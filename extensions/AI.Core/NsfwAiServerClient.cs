using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AI.Core;

public interface INsfwAiServerClient
{
    Task<IReadOnlyList<AiModelCatalogEntry>> GetModelCatalogAsync(AiCoreConnectionSettings settings, CancellationToken ct = default);

    Task<IReadOnlyList<AiModelCatalogEntry>> GetLoadedModelsAsync(AiCoreConnectionSettings settings, CancellationToken ct = default);

    Task<IReadOnlyList<AiModelCatalogEntry>> LoadModelsAsync(AiCoreConnectionSettings settings, AiModelSelectionRequest request, CancellationToken ct = default);

    Task<IReadOnlyList<AiModelCatalogEntry>> UnloadModelsAsync(AiCoreConnectionSettings settings, AiModelSelectionRequest request, CancellationToken ct = default);

    Task<AiCustomPipelineSyncResponse> RegisterCustomPipelineAsync(AiCoreConnectionSettings settings, AiCustomPipelineDefinition pipeline, CancellationToken ct = default);

    Task<AiCustomPipelineSyncResponse> DeleteCustomPipelineAsync(AiCoreConnectionSettings settings, string pipelineName, CancellationToken ct = default);

    Task<JsonElement> AnalyzeImagesAsync(AiCoreConnectionSettings settings, ImageAnalyzeRequest request, CancellationToken ct = default);

    Task<JsonElement> AnalyzeVideoAsync(AiCoreConnectionSettings settings, VideoAnalyzeRequest request, CancellationToken ct = default);

    Task<JsonElement> AnalyzeAudioAsync(AiCoreConnectionSettings settings, AudioAnalyzeRequest request, CancellationToken ct = default);

    Task<TextEncodeResponse> EncodeTextAsync(AiCoreConnectionSettings settings, TextEncodeRequest request, CancellationToken ct = default);
}

public sealed class NsfwAiServerClient(HttpClient httpClient, IAiModelCatalogCache catalogCache) : INsfwAiServerClient
{
    private static readonly JsonSerializerOptions SnakeCaseJson = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DictionaryKeyPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    private readonly HttpClient _httpClient = httpClient;
    private readonly IAiModelCatalogCache _catalogCache = catalogCache;

    public Task<IReadOnlyList<AiModelCatalogEntry>> GetModelCatalogAsync(AiCoreConnectionSettings settings, CancellationToken ct = default)
    {
        var normalized = settings.Normalize();
        return _catalogCache.GetOrCreateAsync(
            normalized.ServerBaseUrl,
            AiModelCatalogCacheKind.Catalog,
            TimeSpan.FromSeconds(normalized.ModelCacheSeconds),
            async innerCt =>
            {
                var envelope = await GetAsync<AiModelCatalogEnvelope>(normalized, "/v4/models/catalog", innerCt);
                return envelope.Models;
            },
            ct);
    }

    public Task<IReadOnlyList<AiModelCatalogEntry>> GetLoadedModelsAsync(AiCoreConnectionSettings settings, CancellationToken ct = default)
    {
        var normalized = settings.Normalize();
        return _catalogCache.GetOrCreateAsync(
            normalized.ServerBaseUrl,
            AiModelCatalogCacheKind.Loaded,
            TimeSpan.FromSeconds(normalized.ModelCacheSeconds),
            async innerCt =>
            {
                var envelope = await GetAsync<AiModelCatalogEnvelope>(normalized, "/v4/models/loaded", innerCt);
                return envelope.Models;
            },
            ct);
    }

    public async Task<IReadOnlyList<AiModelCatalogEntry>> LoadModelsAsync(AiCoreConnectionSettings settings, AiModelSelectionRequest request, CancellationToken ct = default)
    {
        var normalized = settings.Normalize();
        var envelope = await PostAsync<AiModelSelectionRequest, AiModelCatalogEnvelope>(normalized, "/v4/models/load", request, ct);
        _catalogCache.Invalidate(normalized.ServerBaseUrl);
        return envelope.Models;
    }

    public async Task<IReadOnlyList<AiModelCatalogEntry>> UnloadModelsAsync(AiCoreConnectionSettings settings, AiModelSelectionRequest request, CancellationToken ct = default)
    {
        var normalized = settings.Normalize();
        var envelope = await PostAsync<AiModelSelectionRequest, AiModelCatalogEnvelope>(normalized, "/v4/models/unload", request, ct);
        _catalogCache.Invalidate(normalized.ServerBaseUrl);
        return envelope.Models;
    }

    public Task<AiCustomPipelineSyncResponse> RegisterCustomPipelineAsync(AiCoreConnectionSettings settings, AiCustomPipelineDefinition pipeline, CancellationToken ct = default)
        => PostAsync<AiCustomPipelineDefinition, AiCustomPipelineSyncResponse>(settings, "/v4/pipelines/custom", pipeline.Normalize(), ct);

    public Task<AiCustomPipelineSyncResponse> DeleteCustomPipelineAsync(AiCoreConnectionSettings settings, string pipelineName, CancellationToken ct = default)
        => DeleteAsync<AiCustomPipelineSyncResponse>(settings, $"/v4/pipelines/custom/{Uri.EscapeDataString(pipelineName)}", ct);

    public Task<JsonElement> AnalyzeImagesAsync(AiCoreConnectionSettings settings, ImageAnalyzeRequest request, CancellationToken ct = default)
        => PostJsonAsync(settings, "/v4/analyze/images", request, ct);

    public Task<JsonElement> AnalyzeVideoAsync(AiCoreConnectionSettings settings, VideoAnalyzeRequest request, CancellationToken ct = default)
        => PostJsonAsync(settings, "/v4/analyze/video", request, ct);

    public Task<JsonElement> AnalyzeAudioAsync(AiCoreConnectionSettings settings, AudioAnalyzeRequest request, CancellationToken ct = default)
        => PostJsonAsync(settings, "/v4/analyze/audio", request, ct);

    public Task<TextEncodeResponse> EncodeTextAsync(AiCoreConnectionSettings settings, TextEncodeRequest request, CancellationToken ct = default)
        => PostAsync<TextEncodeRequest, TextEncodeResponse>(settings, "/v4/encode/text", request, ct);

    private async Task<TResponse> GetAsync<TResponse>(AiCoreConnectionSettings settings, string relativePath, CancellationToken ct)
    {
        var normalized = settings.Normalize();
        using var timeout = CreateTimeoutSource(normalized, ct);
        using var request = new HttpRequestMessage(HttpMethod.Get, BuildUri(normalized, relativePath));
        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
        await EnsureSuccessAsync(response, relativePath, timeout.Token);
        await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token);
        var payload = await JsonSerializer.DeserializeAsync<TResponse>(stream, SnakeCaseJson, timeout.Token);
        return payload ?? throw new InvalidOperationException($"The AI server returned an empty response for '{relativePath}'.");
    }

    private async Task<TResponse> PostAsync<TRequest, TResponse>(AiCoreConnectionSettings settings, string relativePath, TRequest payload, CancellationToken ct)
    {
        var normalized = settings.Normalize();
        using var timeout = CreateTimeoutSource(normalized, ct);
        using var request = new HttpRequestMessage(HttpMethod.Post, BuildUri(normalized, relativePath))
        {
            Content = JsonContent.Create(payload, options: SnakeCaseJson),
        };
        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
        await EnsureSuccessAsync(response, relativePath, timeout.Token);
        await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token);
        var result = await JsonSerializer.DeserializeAsync<TResponse>(stream, SnakeCaseJson, timeout.Token);
        return result ?? throw new InvalidOperationException($"The AI server returned an empty response for '{relativePath}'.");
    }

    private async Task<TResponse> DeleteAsync<TResponse>(AiCoreConnectionSettings settings, string relativePath, CancellationToken ct)
    {
        var normalized = settings.Normalize();
        using var timeout = CreateTimeoutSource(normalized, ct);
        using var request = new HttpRequestMessage(HttpMethod.Delete, BuildUri(normalized, relativePath));
        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
        await EnsureSuccessAsync(response, relativePath, timeout.Token);
        await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token);
        var result = await JsonSerializer.DeserializeAsync<TResponse>(stream, SnakeCaseJson, timeout.Token);
        return result ?? throw new InvalidOperationException($"The AI server returned an empty response for '{relativePath}'.");
    }

    private async Task<JsonElement> PostJsonAsync<TRequest>(AiCoreConnectionSettings settings, string relativePath, TRequest payload, CancellationToken ct)
    {
        var normalized = settings.Normalize();
        using var timeout = CreateTimeoutSource(normalized, ct);
        using var request = new HttpRequestMessage(HttpMethod.Post, BuildUri(normalized, relativePath))
        {
            Content = JsonContent.Create(payload, options: SnakeCaseJson),
        };
        using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
        await EnsureSuccessAsync(response, relativePath, timeout.Token);
        await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: timeout.Token);
        return document.RootElement.Clone();
    }

    private static Uri BuildUri(AiCoreConnectionSettings settings, string relativePath)
    {
        var baseUrl = settings.ServerBaseUrl.TrimEnd('/');
        var suffix = relativePath.StartsWith("/", StringComparison.Ordinal) ? relativePath : $"/{relativePath}";
        return new Uri($"{baseUrl}{suffix}", UriKind.Absolute);
    }

    private static CancellationTokenSource CreateTimeoutSource(AiCoreConnectionSettings settings, CancellationToken ct)
    {
        var source = CancellationTokenSource.CreateLinkedTokenSource(ct);
        if (settings.RequestTimeoutSeconds > 0)
        {
            source.CancelAfter(TimeSpan.FromSeconds(settings.RequestTimeoutSeconds));
        }

        return source;
    }

    private static async Task EnsureSuccessAsync(HttpResponseMessage response, string relativePath, CancellationToken ct)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        var detail = await response.Content.ReadAsStringAsync(ct);
        if (string.IsNullOrWhiteSpace(detail))
        {
            detail = response.ReasonPhrase ?? "The AI server returned an error.";
        }

        throw new HttpRequestException(
            $"Request to '{relativePath}' failed with {(int)response.StatusCode} ({response.StatusCode}): {detail}",
            null,
            response.StatusCode);
    }
}

using System.Text.Json;

using Cove.Core.Interfaces;
using Cove.Plugins;

using Pgvector;

namespace AI.Core;

internal sealed class AiCoreSemanticTextEncoder(
    IExtensionStoreFactory storeFactory,
    INsfwAiServerClient aiServerClient) : ITextEncoder
{
    private const string ExtensionId = "cove.community.ai.core";
    private const string SettingsStoreKey = "settings";

    private static readonly JsonSerializerOptions StateJson = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    private readonly IExtensionStoreFactory _storeFactory = storeFactory;
    private readonly INsfwAiServerClient _aiServerClient = aiServerClient;

    public string KindFamily => "semantic.v1";

    public async Task<Vector> EncodeAsync(string text, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException("Text is required.", nameof(text));
        }

        var settings = await LoadSettingsAsync(cancellationToken);
        var response = await _aiServerClient.EncodeTextAsync(
            settings,
            new TextEncodeRequest
            {
                Text = text.Trim(),
                KindFamily = KindFamily,
            },
            cancellationToken);

        if (response.Vector.Count == 0)
        {
            throw new InvalidOperationException($"The AI server returned an empty text embedding for kind family '{KindFamily}'.");
        }

        if (response.Dim > 0 && response.Dim != response.Vector.Count)
        {
            throw new InvalidOperationException(
                $"The AI server returned a text embedding with reported dimension {response.Dim} but vector length {response.Vector.Count}.");
        }

        return new Vector(response.Vector.ToArray());
    }

    private async Task<AiCoreConnectionSettings> LoadSettingsAsync(CancellationToken cancellationToken)
    {
        var store = _storeFactory.CreateStore(ExtensionId);
        var payload = await store.GetAsync(SettingsStoreKey, cancellationToken);

        if (string.IsNullOrWhiteSpace(payload))
        {
            return new AiCoreConnectionSettings().Normalize();
        }

        var settings = JsonSerializer.Deserialize<AiCoreConnectionSettings>(payload, StateJson) ?? new AiCoreConnectionSettings();
        return settings.Normalize();
    }
}

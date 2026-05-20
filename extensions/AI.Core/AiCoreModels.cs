using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

using AI.Extensions.Abstractions;

namespace AI.Core;

public sealed record AiCoreConnectionSettings
{
    public const int DefaultRequestTimeoutSeconds = 900;

    public const int LegacyDefaultRequestTimeoutSeconds = 120;

    public string ServerBaseUrl { get; init; } = "http://127.0.0.1:8000";

    public string DefaultLoadPolicy { get; init; } = AiLoadPolicies.LoadOrFail;

    public double? DefaultThreshold { get; init; }

    public int RequestTimeoutSeconds { get; init; } = DefaultRequestTimeoutSeconds;

    public int MaxInFlight { get; init; } = 2;

    public bool DispatchResultsByDefault { get; init; } = true;

    public List<AiPathMapping> PathMappings { get; init; } = [];

    public List<AiTaggingModelPreference> TaggingModelPreferences { get; init; } = [];

    public List<AiModelSupersessionRule> ModelSupersessions { get; init; } = [];

    public AiCoreConnectionSettings Normalize()
    {
        var trimmedBaseUrl = (ServerBaseUrl ?? string.Empty).Trim();
        if (string.IsNullOrWhiteSpace(trimmedBaseUrl))
        {
            throw new ArgumentException("ServerBaseUrl is required.", nameof(ServerBaseUrl));
        }

        if (!Uri.TryCreate(trimmedBaseUrl, UriKind.Absolute, out var baseUri) ||
            (baseUri.Scheme != Uri.UriSchemeHttp && baseUri.Scheme != Uri.UriSchemeHttps))
        {
            throw new ArgumentException("ServerBaseUrl must be an absolute HTTP or HTTPS URL.", nameof(ServerBaseUrl));
        }

        var normalizedLoadPolicy = string.IsNullOrWhiteSpace(DefaultLoadPolicy)
            ? AiLoadPolicies.LoadOrFail
            : DefaultLoadPolicy.Trim();

        if (!AiLoadPolicies.All.Contains(normalizedLoadPolicy))
        {
            throw new ArgumentException(
                $"DefaultLoadPolicy must be one of: {string.Join(", ", AiLoadPolicies.All.OrderBy(static item => item))}.",
                nameof(DefaultLoadPolicy));
        }

        var normalizedMappings = (PathMappings ?? [])
            .Where(static mapping => !string.IsNullOrWhiteSpace(mapping.FromPrefix) && !string.IsNullOrWhiteSpace(mapping.ToPrefix))
            .Select(static mapping => mapping.Normalize())
            .DistinctBy(static mapping => mapping.FromPrefix, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var normalizedTaggingPreferences = (TaggingModelPreferences ?? [])
            .Where(static preference => !string.IsNullOrWhiteSpace(preference.Scope)
                && !string.IsNullOrWhiteSpace(preference.Category)
                && !string.IsNullOrWhiteSpace(preference.Model))
            .Select(static preference => preference.Normalize())
            .DistinctBy(static preference => $"{preference.Scope}\u001F{preference.Category}", StringComparer.OrdinalIgnoreCase)
            .ToList();

        var normalizedRequestTimeoutSeconds = RequestTimeoutSeconds switch
        {
            < 0 => 0,
            LegacyDefaultRequestTimeoutSeconds => DefaultRequestTimeoutSeconds,
            _ => RequestTimeoutSeconds,
        };

        var normalizedModelSupersessions = (ModelSupersessions ?? [])
            .Where(static rule => !string.IsNullOrWhiteSpace(rule.Capability)
                && !string.IsNullOrWhiteSpace(rule.Scope)
                && rule.Models is { Count: > 1 })
            .Select(static rule => rule.Normalize())
            .DistinctBy(static rule => $"{rule.Capability}\u001F{rule.Scope}\u001F{rule.Category ?? string.Empty}", StringComparer.OrdinalIgnoreCase)
            .ToList();

        return this with
        {
            ServerBaseUrl = baseUri.ToString().TrimEnd('/'),
            DefaultLoadPolicy = normalizedLoadPolicy,
            RequestTimeoutSeconds = normalizedRequestTimeoutSeconds,
            MaxInFlight = Math.Max(1, MaxInFlight),
            PathMappings = normalizedMappings,
            TaggingModelPreferences = normalizedTaggingPreferences,
            ModelSupersessions = normalizedModelSupersessions,
        };
    }
}

public sealed record AiModelSupersessionRule
{
    public string Capability { get; init; } = string.Empty;

    public string Scope { get; init; } = string.Empty;

    public string? Category { get; init; }

    public List<string> Models { get; init; } = [];

    public AiModelSupersessionRule Normalize()
    {
        return this with
        {
            Capability = (Capability ?? string.Empty).Trim().ToLowerInvariant(),
            Scope = (Scope ?? string.Empty).Trim().ToLowerInvariant(),
            Category = string.IsNullOrWhiteSpace(Category) ? null : Category.Trim(),
            Models = (Models ?? [])
                .Where(static model => !string.IsNullOrWhiteSpace(model))
                .Select(static model => model.Trim())
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList(),
        };
    }
}

public sealed record AiTaggingModelPreference
{
    public string Scope { get; init; } = string.Empty;

    public string Category { get; init; } = string.Empty;

    public string Model { get; init; } = string.Empty;

    public AiTaggingModelPreference Normalize()
    {
        return this with
        {
            Scope = (Scope ?? string.Empty).Trim().ToLowerInvariant(),
            Category = (Category ?? string.Empty).Trim(),
            Model = (Model ?? string.Empty).Trim(),
        };
    }
}

public sealed record AiPathMapping
{
    public string FromPrefix { get; init; } = string.Empty;

    public string ToPrefix { get; init; } = string.Empty;

    public AiPathMapping Normalize()
    {
        return this with
        {
            FromPrefix = NormalizePrefix(FromPrefix),
            ToPrefix = NormalizePrefix(ToPrefix),
        };
    }

    private static string NormalizePrefix(string prefix)
    {
        var trimmed = (prefix ?? string.Empty).Trim();
        if (trimmed.Length <= 1)
        {
            return trimmed;
        }

        return trimmed.TrimEnd('\\', '/');
    }
}

public sealed class AiModelCatalogEnvelope
{
    public List<AiModelCatalogEntry> Models { get; init; } = [];
}

public sealed class AiModelCatalogEntry
{
    public string ConfigName { get; init; } = string.Empty;

    public string Name { get; init; } = string.Empty;

    public int? Identifier { get; init; }

    [JsonConverter(typeof(FlexibleStringJsonConverter))]
    public string? Version { get; init; }

    public List<string> Categories { get; init; } = [];

    public string? Type { get; init; }

    public List<string> Capabilities { get; init; } = [];

    public List<string> SupportedScopes { get; init; } = [];

    public bool Active { get; init; }

    public bool Loaded { get; init; }

    public bool Pinned { get; init; }
}

public sealed class FlexibleStringJsonConverter : JsonConverter<string?>
{
    public override string? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Null)
        {
            return null;
        }

        using var document = JsonDocument.ParseValue(ref reader);
        return document.RootElement.ValueKind == JsonValueKind.Null
            ? null
            : document.RootElement.ToString();
    }

    public override void Write(Utf8JsonWriter writer, string? value, JsonSerializerOptions options)
    {
        if (value is null)
        {
            writer.WriteNullValue();
            return;
        }

        writer.WriteStringValue(value);
    }
}

public class AiModelSelectionRequest
{
    public List<string> Models { get; init; } = [];
}

public sealed class AiModelPinRequest : AiModelSelectionRequest
{
    public bool Pinned { get; init; } = true;
}

public sealed class AnalyzeWantRequest
{
    public string? Capability { get; init; }

    public List<string>? Capabilities { get; init; }

    public string? Scope { get; init; }

    public List<string>? Scopes { get; init; }

    public List<string>? Models { get; init; }

    public string? FromDetection { get; init; }
}

public sealed class ImageAnalyzeRequest
{
    public List<string> Paths { get; init; } = [];

    public double? Threshold { get; init; }

    public bool ReturnConfidence { get; init; } = true;

    public List<string>? CategoriesToSkip { get; init; }

    public List<AnalyzeWantRequest>? Want { get; init; }

    public string LoadPolicy { get; init; } = AiLoadPolicies.LoadOrFail;
}

public sealed class VideoAnalyzeRequest
{
    public string Path { get; init; } = string.Empty;

    public double? FrameInterval { get; init; }

    public double? Threshold { get; init; }

    public bool ReturnConfidence { get; init; } = true;

    public bool VrVideo { get; init; }

    public List<string>? CategoriesToSkip { get; init; }

    public List<AnalyzeWantRequest>? Want { get; init; }

    public string LoadPolicy { get; init; } = AiLoadPolicies.LoadOrFail;
}

public sealed class AudioAnalyzeRequest
{
    public List<string> Paths { get; init; } = [];

    public double? Threshold { get; init; }

    public List<AnalyzeWantRequest>? Want { get; init; }

    public string LoadPolicy { get; init; } = AiLoadPolicies.LoadOrFail;
}

public sealed class TextEncodeRequest
{
    public string Text { get; init; } = string.Empty;

    public string KindFamily { get; init; } = string.Empty;
}

public sealed class TextEncodeResponse
{
    public List<float> Vector { get; init; } = [];

    public int Dim { get; init; }

    public string? ModelKey { get; init; }
}

public sealed class AiRunImagesRequest
{
    public List<string> Paths { get; init; } = [];

    public string? EntityType { get; init; }

    public int? EntityId { get; init; }

    public List<string>? ClaimIds { get; init; }

    public double? Threshold { get; init; }

    public bool? ReturnConfidence { get; init; }

    public List<string>? CategoriesToSkip { get; init; }

    public string? LoadPolicy { get; init; }

    public bool? DispatchResults { get; init; }

    public List<string>? ForceClaimIds { get; init; }
}

public sealed class AiRunVideoRequest
{
    public string Path { get; init; } = string.Empty;

    public string? EntityType { get; init; }

    public int? EntityId { get; init; }

    public List<string>? ClaimIds { get; init; }

    public double? FrameInterval { get; init; }

    public double? Threshold { get; init; }

    public bool? ReturnConfidence { get; init; }

    public bool VrVideo { get; init; }

    public List<string>? CategoriesToSkip { get; init; }

    public string? LoadPolicy { get; init; }

    public bool? DispatchResults { get; init; }

    public List<string>? ForceClaimIds { get; init; }
}

public sealed class AiRunAudioRequest
{
    public List<string> Paths { get; init; } = [];

    public string? EntityType { get; init; }

    public int? EntityId { get; init; }

    public List<string>? ClaimIds { get; init; }

    public double? Threshold { get; init; }

    public string? LoadPolicy { get; init; }

    public bool? DispatchResults { get; init; }

    public List<string>? ForceClaimIds { get; init; }
}

public enum AiRunPlanDecision
{
    Skip = 1,
    Run = 2,
    Rerun = 3,
}

public sealed record AiRunPlanItem(
    string ClaimId,
    string ExtensionId,
    string Capability,
    string Scope,
    IReadOnlyList<string> DesiredModels,
    IReadOnlyList<string> ExecutionModels,
    AiRunPlanDecision Decision,
    IReadOnlyList<string> Reasons,
    bool Forced);

public sealed record AiRunResponse(
    string RunId,
    string MediaKind,
    IReadOnlyList<AiCapabilityClaim> Claims,
    JsonElement Analysis,
    IReadOnlyList<AiDispatchResult> DispatchResults,
    IReadOnlyList<AiRunPlanItem> Plan
);

public sealed record AiQueueRunRequest
{
    public string MediaKind { get; init; } = AiMediaKinds.Video;

    public string? EntityType { get; init; }

    public List<int> EntityIds { get; init; } = [];

    public List<string> Paths { get; init; } = [];

    public List<string>? ClaimIds { get; init; }

    public double? FrameInterval { get; init; }

    public double? Threshold { get; init; }

    public bool? ReturnConfidence { get; init; }

    public bool VrVideo { get; init; }

    public List<string>? CategoriesToSkip { get; init; }

    public string? LoadPolicy { get; init; }

    public bool? DispatchResults { get; init; }

    public List<string>? ForceClaimIds { get; init; }

    public AiQueueRunRequest Normalize()
    {
        var normalizedEntityType = NormalizeEntityType(EntityType);
        var inferredMediaKind = InferMediaKind(normalizedEntityType);
        var normalizedMediaKind = string.IsNullOrWhiteSpace(MediaKind)
            ? inferredMediaKind ?? AiMediaKinds.Video
            : MediaKind.Trim().ToLowerInvariant();

        if (normalizedMediaKind is not (AiMediaKinds.Image or AiMediaKinds.Video or AiMediaKinds.Audio))
        {
            throw new ArgumentException($"Unsupported media kind '{MediaKind}'.", nameof(MediaKind));
        }

        var normalizedIds = (EntityIds ?? [])
            .Where(static id => id > 0)
            .Distinct()
            .ToList();

        var normalizedPaths = (Paths ?? [])
            .Where(static path => !string.IsNullOrWhiteSpace(path))
            .Select(static path => path.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (normalizedIds.Count == 0 && normalizedPaths.Count == 0)
        {
            throw new ArgumentException("At least one path or entity id is required.", nameof(Paths));
        }

        if (normalizedIds.Count > 0 && normalizedEntityType is null)
        {
            throw new ArgumentException("EntityType is required when EntityIds are supplied.", nameof(EntityType));
        }

        if (normalizedEntityType == "image" && normalizedMediaKind != AiMediaKinds.Image)
        {
            throw new ArgumentException("Image selections can only be queued with media kind 'image'.", nameof(MediaKind));
        }

        if (normalizedEntityType == "scene" && normalizedMediaKind == AiMediaKinds.Image)
        {
            throw new ArgumentException("Scene selections do not support media kind 'image'.", nameof(MediaKind));
        }

        var normalizedClaimIds = NormalizeStringList(ClaimIds);
        var normalizedForceClaimIds = NormalizeStringList(ForceClaimIds);
        var normalizedCategoriesToSkip = NormalizeStringList(CategoriesToSkip);

        return this with
        {
            MediaKind = normalizedMediaKind,
            EntityType = normalizedEntityType,
            EntityIds = normalizedIds,
            Paths = normalizedPaths,
            ClaimIds = normalizedClaimIds.Count == 0 ? null : normalizedClaimIds,
            ForceClaimIds = normalizedForceClaimIds.Count == 0 ? null : normalizedForceClaimIds,
            CategoriesToSkip = normalizedCategoriesToSkip.Count == 0 ? null : normalizedCategoriesToSkip,
            LoadPolicy = string.IsNullOrWhiteSpace(LoadPolicy) ? null : LoadPolicy.Trim(),
        };
    }

    public static AiQueueRunRequest FromJobParameters(IReadOnlyDictionary<string, string>? parameters)
    {
        if (parameters is null || parameters.Count == 0)
        {
            throw new ArgumentException("AI job parameters are required.", nameof(parameters));
        }

        return new AiQueueRunRequest
        {
            MediaKind = GetValue(parameters, "mediaKind") ?? GetValue(parameters, "media_kind") ?? AiMediaKinds.Video,
            EntityType = GetValue(parameters, "entityType") ?? GetValue(parameters, "entity_type"),
            EntityIds = ParseIntList(GetValue(parameters, "entityIds") ?? GetValue(parameters, "entity_ids") ?? GetValue(parameters, "ids")),
            Paths = ParseDelimitedList(GetValue(parameters, "paths") ?? GetValue(parameters, "path")),
            ClaimIds = ParseDelimitedList(GetValue(parameters, "claimIds") ?? GetValue(parameters, "claim_ids")),
            ForceClaimIds = ParseDelimitedList(GetValue(parameters, "forceClaimIds") ?? GetValue(parameters, "force_claim_ids")),
            FrameInterval = ParseNullableDouble(GetValue(parameters, "frameInterval") ?? GetValue(parameters, "frame_interval")),
            Threshold = ParseNullableDouble(GetValue(parameters, "threshold")),
            ReturnConfidence = ParseNullableBool(GetValue(parameters, "returnConfidence") ?? GetValue(parameters, "return_confidence")),
            VrVideo = ParseNullableBool(GetValue(parameters, "vrVideo") ?? GetValue(parameters, "vr_video")) ?? false,
            CategoriesToSkip = ParseDelimitedList(GetValue(parameters, "categoriesToSkip") ?? GetValue(parameters, "categories_to_skip")),
            LoadPolicy = GetValue(parameters, "loadPolicy") ?? GetValue(parameters, "load_policy"),
            DispatchResults = ParseNullableBool(GetValue(parameters, "dispatchResults") ?? GetValue(parameters, "dispatch_results")),
        }.Normalize();
    }

    public static AiQueueRunRequest FromActionPayload(JsonElement payload)
    {
        if (payload.ValueKind != JsonValueKind.Object)
        {
            throw new ArgumentException("AI action payload must be a JSON object.", nameof(payload));
        }

        return new AiQueueRunRequest
        {
            MediaKind = ReadString(payload, "mediaKind", "media_kind")
                ?? InferMediaKind(NormalizeEntityType(ReadString(payload, "entityType", "entity_type", "pageType", "page_type")))
                ?? AiMediaKinds.Video,
            EntityType = ReadString(payload, "entityType", "entity_type", "pageType", "page_type"),
            EntityIds = ReadIntList(payload, "entityIds", "entity_ids", "selectedIds", "selected_ids", "ids", "entityId", "entity_id"),
            Paths = ReadStringList(payload, "paths", "path"),
            ClaimIds = ReadStringList(payload, "claimIds", "claim_ids"),
            ForceClaimIds = ReadStringList(payload, "forceClaimIds", "force_claim_ids"),
            FrameInterval = ReadNullableDouble(payload, "frameInterval", "frame_interval"),
            Threshold = ReadNullableDouble(payload, "threshold"),
            ReturnConfidence = ReadNullableBool(payload, "returnConfidence", "return_confidence"),
            VrVideo = ReadNullableBool(payload, "vrVideo", "vr_video") ?? false,
            CategoriesToSkip = ReadStringList(payload, "categoriesToSkip", "categories_to_skip"),
            LoadPolicy = ReadString(payload, "loadPolicy", "load_policy"),
            DispatchResults = ReadNullableBool(payload, "dispatchResults", "dispatch_results"),
        }.Normalize();
    }

    private static string? GetValue(IReadOnlyDictionary<string, string> parameters, string key)
        => parameters.TryGetValue(key, out var value) ? value : null;

    private static List<string> NormalizeStringList(List<string>? values)
        => (values ?? [])
            .Where(static value => !string.IsNullOrWhiteSpace(value))
            .Select(static value => value.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

    private static List<string> ParseDelimitedList(string? raw)
        => string.IsNullOrWhiteSpace(raw)
            ? []
            : raw.Split([',', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Where(static item => !string.IsNullOrWhiteSpace(item))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

    private static List<int> ParseIntList(string? raw)
        => ParseDelimitedList(raw)
            .Select(static item => int.TryParse(item, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) ? value : (int?)null)
            .Where(static value => value.HasValue && value.Value > 0)
            .Select(static value => value!.Value)
            .Distinct()
            .ToList();

    private static double? ParseNullableDouble(string? raw)
        => double.TryParse(raw, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var value)
            ? value
            : null;

    private static bool? ParseNullableBool(string? raw)
        => bool.TryParse(raw, out var value)
            ? value
            : null;

    private static string? NormalizeEntityType(string? entityType)
    {
        var normalized = entityType?.Trim().ToLowerInvariant();
        return normalized switch
        {
            "scene" or "scenes" => "scene",
            "image" or "images" => "image",
            null or "" => null,
            _ => normalized,
        };
    }

    private static string? InferMediaKind(string? entityType)
        => entityType switch
        {
            "image" => AiMediaKinds.Image,
            "scene" => AiMediaKinds.Video,
            _ => null,
        };

    private static string? ReadString(JsonElement payload, params string[] propertyNames)
    {
        foreach (var propertyName in propertyNames)
        {
            if (!TryGetPropertyIgnoreCase(payload, propertyName, out var property))
            {
                continue;
            }

            if (property.ValueKind == JsonValueKind.String)
            {
                var value = property.GetString();
                if (!string.IsNullOrWhiteSpace(value))
                {
                    return value.Trim();
                }
            }

            if (property.ValueKind is JsonValueKind.Number or JsonValueKind.True or JsonValueKind.False)
            {
                return property.ToString();
            }
        }

        return null;
    }

    private static List<string> ReadStringList(JsonElement payload, params string[] propertyNames)
    {
        foreach (var propertyName in propertyNames)
        {
            if (!TryGetPropertyIgnoreCase(payload, propertyName, out var property))
            {
                continue;
            }

            if (property.ValueKind == JsonValueKind.Array)
            {
                return property.EnumerateArray()
                    .Select(static item => item.ValueKind == JsonValueKind.String ? item.GetString() : item.ToString())
                    .Where(static item => !string.IsNullOrWhiteSpace(item))
                    .Select(static item => item!.Trim())
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();
            }

            if (property.ValueKind == JsonValueKind.String)
            {
                return ParseDelimitedList(property.GetString());
            }
        }

        return [];
    }

    private static List<int> ReadIntList(JsonElement payload, params string[] propertyNames)
    {
        foreach (var propertyName in propertyNames)
        {
            if (!TryGetPropertyIgnoreCase(payload, propertyName, out var property))
            {
                continue;
            }

            if (property.ValueKind == JsonValueKind.Array)
            {
                return property.EnumerateArray()
                    .Select(static item => item.ValueKind == JsonValueKind.Number && item.TryGetInt32(out var numericValue)
                        ? numericValue
                        : int.TryParse(item.ToString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedValue)
                            ? parsedValue
                            : (int?)null)
                    .Where(static value => value.HasValue && value.Value > 0)
                    .Select(static value => value!.Value)
                    .Distinct()
                    .ToList();
            }

            if (property.ValueKind is JsonValueKind.Number or JsonValueKind.String)
            {
                return ParseIntList(property.ToString());
            }
        }

        return [];
    }

    private static double? ReadNullableDouble(JsonElement payload, params string[] propertyNames)
    {
        foreach (var propertyName in propertyNames)
        {
            if (!TryGetPropertyIgnoreCase(payload, propertyName, out var property))
            {
                continue;
            }

            if (property.ValueKind == JsonValueKind.Number && property.TryGetDouble(out var numericValue))
            {
                return numericValue;
            }

            if (property.ValueKind == JsonValueKind.String)
            {
                return ParseNullableDouble(property.GetString());
            }
        }

        return null;
    }

    private static bool? ReadNullableBool(JsonElement payload, params string[] propertyNames)
    {
        foreach (var propertyName in propertyNames)
        {
            if (!TryGetPropertyIgnoreCase(payload, propertyName, out var property))
            {
                continue;
            }

            if (property.ValueKind == JsonValueKind.True)
            {
                return true;
            }

            if (property.ValueKind == JsonValueKind.False)
            {
                return false;
            }

            if (property.ValueKind == JsonValueKind.String)
            {
                return ParseNullableBool(property.GetString());
            }
        }

        return null;
    }

    private static bool TryGetPropertyIgnoreCase(JsonElement payload, string propertyName, out JsonElement property)
    {
        foreach (var candidate in payload.EnumerateObject())
        {
            if (string.Equals(candidate.Name, propertyName, StringComparison.OrdinalIgnoreCase))
            {
                property = candidate.Value;
                return true;
            }
        }

        property = default;
        return false;
    }
}

public sealed record AiResolvedRunTarget(
    string UnitId,
    string Label,
    string Path,
    int? EntityId = null,
    string? EntityType = null
);

public sealed record AiQueuedRunResponse(
    string JobId,
    string Description,
    int TargetCount,
    string MediaKind,
    IReadOnlyList<string> ClaimIds
);

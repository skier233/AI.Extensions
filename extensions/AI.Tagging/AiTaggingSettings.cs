using System.Text.Json;

using Cove.Plugins;

using Microsoft.Extensions.DependencyInjection;

namespace AI.Tagging;

internal sealed record AiTaggingSettings
{
    public List<AiTagNameOverride> TagNameOverrides { get; init; } = [];

    public AiTaggingSettings Normalize()
        => this with
        {
            TagNameOverrides = (TagNameOverrides ?? [])
                .Where(static item => !string.IsNullOrWhiteSpace(item.SourceTagName) && !string.IsNullOrWhiteSpace(item.TargetTagName))
                .Select(static item => item.Normalize())
                .DistinctBy(static item => item.SourceTagName, StringComparer.OrdinalIgnoreCase)
                .ToList(),
        };

    public IReadOnlyDictionary<string, string> ToOverrideMap()
        => Normalize().TagNameOverrides.ToDictionary(
            static item => item.SourceTagName,
            static item => item.TargetTagName,
            StringComparer.OrdinalIgnoreCase);
}

internal sealed record AiTagNameOverride
{
    public string SourceTagName { get; init; } = string.Empty;

    public string TargetTagName { get; init; } = string.Empty;

    public AiTagNameOverride Normalize()
        => this with
        {
            SourceTagName = (SourceTagName ?? string.Empty).Trim(),
            TargetTagName = (TargetTagName ?? string.Empty).Trim(),
        };
}

internal static class AiTaggingSettingsStore
{
    public const string ExtensionId = "cove.ai.tagging";

    public const string SettingsStoreKey = "settings";

    private static readonly JsonSerializerOptions StateJson = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    public static async Task<AiTaggingSettings> LoadAsync(IExtensionStore store, CancellationToken ct = default)
    {
        var payload = await store.GetAsync(SettingsStoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
        {
            return new AiTaggingSettings().Normalize();
        }

        var settings = JsonSerializer.Deserialize<AiTaggingSettings>(payload, StateJson) ?? new AiTaggingSettings();
        return settings.Normalize();
    }

    public static async Task<AiTaggingSettings> LoadAsync(IServiceProvider services, CancellationToken ct = default)
    {
        var storeFactory = services.GetService<IExtensionStoreFactory>();
        if (storeFactory is null)
        {
            return new AiTaggingSettings().Normalize();
        }

        return await LoadAsync(storeFactory.CreateStore(ExtensionId), ct);
    }

    public static Task SaveAsync(IExtensionStore store, AiTaggingSettings settings, CancellationToken ct = default)
        => store.SetAsync(SettingsStoreKey, JsonSerializer.Serialize(settings.Normalize(), StateJson), ct);
}
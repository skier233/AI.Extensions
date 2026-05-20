using AI.Core;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiPathMapperTests
{
    [Fact]
    public void MapPath_RewritesConfiguredPrefix()
    {
        var mapped = AiPathMapper.MapPath(
            [new AiPathMapping { FromPrefix = "C:/media", ToPrefix = "/mnt/media" }],
            "C:/media/scenes/example.mp4");

        Assert.Equal("/mnt/media/scenes/example.mp4", mapped);
    }

    [Fact]
    public void MapPath_NormalizesMalformedWindowsDrivePathWithoutMappings()
    {
        var mapped = AiPathMapper.MapPath([], "E:test/Content/Videos/example.mp4");

        Assert.Equal("E:/test/Content/Videos/example.mp4", mapped);
    }

    [Fact]
    public void Normalize_UsesExtendedDefaultRequestTimeout()
    {
        var normalized = new AiCoreConnectionSettings().Normalize();

        Assert.Equal(AiCoreConnectionSettings.DefaultRequestTimeoutSeconds, normalized.RequestTimeoutSeconds);
    }

    [Fact]
    public void Normalize_UpgradesLegacyRequestTimeoutToExtendedDefault()
    {
        var normalized = new AiCoreConnectionSettings
        {
            RequestTimeoutSeconds = AiCoreConnectionSettings.LegacyDefaultRequestTimeoutSeconds,
        }.Normalize();

        Assert.Equal(AiCoreConnectionSettings.DefaultRequestTimeoutSeconds, normalized.RequestTimeoutSeconds);
    }

    [Fact]
    public void Normalize_DeduplicatesTaggingModelPreferencesByScopeAndCategory()
    {
        var normalized = new AiCoreConnectionSettings
        {
            TaggingModelPreferences =
            [
                new AiTaggingModelPreference { Scope = " Asset ", Category = "Actions", Model = "tagger-old" },
                new AiTaggingModelPreference { Scope = "asset", Category = "Actions", Model = "tagger-new" },
                new AiTaggingModelPreference { Scope = "frame", Category = "Actions", Model = "tagger-frame" },
            ],
        }.Normalize();

        Assert.Equal(2, normalized.TaggingModelPreferences.Count);
        Assert.Contains(normalized.TaggingModelPreferences, preference =>
            preference.Scope == "asset"
            && preference.Category == "Actions"
            && preference.Model == "tagger-old");
        Assert.Contains(normalized.TaggingModelPreferences, preference =>
            preference.Scope == "frame"
            && preference.Category == "Actions"
            && preference.Model == "tagger-frame");
    }
}

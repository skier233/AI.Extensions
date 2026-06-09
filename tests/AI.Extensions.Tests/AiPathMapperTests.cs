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
            "C:/media/videos/example.mp4");

        Assert.Equal("/mnt/media/videos/example.mp4", mapped);
    }

    [Fact]
    public void MapPath_MatchesSourcePrefixEnteredWithWindowsBackslashes()
    {
        var mapped = AiPathMapper.MapPath(
            [new AiPathMapping { FromPrefix = "C:\\media", ToPrefix = "/mnt/media" }],
            "C:\\media\\videos\\example.mp4");

        Assert.Equal("/mnt/media/videos/example.mp4", mapped);
    }

    [Fact]
    public void MapPath_DoesNotMatchPartialPathSegment()
    {
        var mapped = AiPathMapper.MapPath(
            [new AiPathMapping { FromPrefix = "C:/media", ToPrefix = "/mnt/media" }],
            "C:/media-extra/example.mp4");

        Assert.Equal("C:/media-extra/example.mp4", mapped);
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
    public void Normalize_DeduplicatesCapabilityModelBindingsByCapabilitySlotScopeAndCategory()
    {
        var normalized = new AiCoreConnectionSettings
        {
            CapabilityModelBindings =
            [
                new AiCapabilityModelBinding { CapabilityId = "Tagging", SlotId = "Category", Scope = " Asset ", Category = "Actions", Model = "tagger-old" },
                new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "asset", Category = "Actions", Model = "tagger-new" },
                new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "frame", Category = "Actions", Model = "tagger-frame" },
            ],
        }.Normalize();

        Assert.Equal(2, normalized.CapabilityModelBindings.Count);
        Assert.Contains(normalized.CapabilityModelBindings, binding =>
            binding.CapabilityId == "tagging"
            && binding.SlotId == "category"
            && binding.Scope == "asset"
            && binding.Category == "Actions"
            && binding.Model == "tagger-old");
        Assert.Contains(normalized.CapabilityModelBindings, binding =>
            binding.CapabilityId == "tagging"
            && binding.SlotId == "category"
            && binding.Scope == "frame"
            && binding.Category == "Actions"
            && binding.Model == "tagger-frame");
    }
}

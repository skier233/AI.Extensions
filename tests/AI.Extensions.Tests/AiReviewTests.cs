using System.Text.Json;

using AI.Review;

using Xunit;

namespace AI.Extensions.Tests;

public class AiReviewFramePlannerTests
{
    [Fact]
    public void Plan_ShortSegment_UsesTheObservedFrame()
    {
        // One observation at 10s with a 2s interval: AI.Tagging stores [10, 12).
        var frames = AiReviewFramePlanner.Plan(10, 12, observationCount: 1, maxFrames: 3);
        Assert.Equal([10.0], frames);
    }

    [Fact]
    public void Plan_MediumSegment_PicksFirstAndLastObservation()
    {
        // Three observations at 10, 12, 14 → segment [10, 16): 6s long → 2 frames.
        var frames = AiReviewFramePlanner.Plan(10, 16, observationCount: 3, maxFrames: 3);
        Assert.Equal([10.0, 14.0], frames);
    }

    [Fact]
    public void Plan_LongSegment_PicksFirstMiddleLast()
    {
        // Ten observations at 0,2,...,18 → segment [0, 20).
        var frames = AiReviewFramePlanner.Plan(0, 20, observationCount: 10, maxFrames: 3);
        Assert.Equal([0.0, 8.0, 18.0], frames);
    }

    [Fact]
    public void Plan_RespectsMaxFrames()
    {
        var frames = AiReviewFramePlanner.Plan(0, 20, observationCount: 10, maxFrames: 1);
        Assert.Single(frames);
        Assert.Equal(8.0, frames[0]);
    }

    [Fact]
    public void Plan_WithoutObservationCount_FallsBackToFractions()
    {
        var frames = AiReviewFramePlanner.Plan(100, 120, observationCount: 0, maxFrames: 3);
        Assert.Equal([105.0, 110.0, 115.0], frames);

        var single = AiReviewFramePlanner.Plan(5, 6, observationCount: 0, maxFrames: 3);
        Assert.Equal([5.5], single);
    }

    [Fact]
    public void Plan_NeverExceedsObservationCount()
    {
        var frames = AiReviewFramePlanner.Plan(0, 30, observationCount: 2, maxFrames: 3);
        Assert.Equal([0.0, 15.0], frames);
    }

    [Fact]
    public void ReadObservationCount_AcceptsStringAndNumber()
    {
        using var asString = JsonDocument.Parse("""{"modelKey":"bodyparts","observationCount":"4"}""");
        using var asNumber = JsonDocument.Parse("""{"observationCount":7}""");
        using var missing = JsonDocument.Parse("""{"modelKey":"x"}""");
        Assert.Equal(4, AiReviewFramePlanner.ReadObservationCount(asString));
        Assert.Equal(7, AiReviewFramePlanner.ReadObservationCount(asNumber));
        Assert.Equal(0, AiReviewFramePlanner.ReadObservationCount(missing));
        Assert.Equal("bodyparts", AiReviewFramePlanner.ReadModelKey(asString));
        Assert.Null(AiReviewFramePlanner.ReadModelKey(null));
    }
}

public class AiReviewExportServiceTests
{
    [Theory]
    [InlineData("Feet", "feet")]
    [InlineData("Pussy (Vis)", "pussy (vis)")]
    [InlineData("Ball Licking/Sucking", "ball licking_sucking")]
    [InlineData("  Tribbing/Scissoring ", "tribbing_scissoring")]
    [InlineData("", "_unknown")]
    public void TagFolderName_IsLowercaseAndFileSystemSafe(string tag, string expected)
    {
        Assert.Equal(expected, AiReviewExportService.TagFolderName(tag));
    }

    [Fact]
    public void Csv_QuotesOnlyWhenNeeded()
    {
        Assert.Equal("plain", AiReviewExportService.Csv("plain"));
        Assert.Equal("\"a,b\"", AiReviewExportService.Csv("a,b"));
        Assert.Equal("\"say \"\"hi\"\"\"", AiReviewExportService.Csv("say \"hi\""));
    }

    [Fact]
    public void ParseFrameSeconds_ReadsMillisecondsFromFileName()
    {
        Assert.Equal(12.345, AiReviewExportService.ParseFrameSeconds(@"C:\export\feet\fp\v12_s99_000012345.jpg"));
        Assert.Null(AiReviewExportService.ParseFrameSeconds(@"C:\export\feet\fp\oops.jpg"));
    }
}

public class AiReviewSettingsTests
{
    [Fact]
    public void Normalize_ClampsAndDefaults()
    {
        var settings = new AiReviewSettings { ThumbnailWidth = 0, MaxFramesPerSegment = 9, SourceKey = " ", ExportRootPath = " D:\\x " }.Normalize();
        Assert.Equal(480, settings.ThumbnailWidth);
        Assert.Equal(3, settings.MaxFramesPerSegment);
        Assert.Equal("ext:ai.tagging", settings.SourceKey);
        Assert.Equal("D:\\x", settings.ExportRootPath);
    }
}

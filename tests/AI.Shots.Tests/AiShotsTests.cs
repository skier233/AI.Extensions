using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

using AI.Extensions.Abstractions;
using AI.Shots;

using Cove.Core.DTOs;
using Cove.Core.Interfaces;
using Cove.Plugins;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;

using Xunit;

namespace AI.Shots.Tests;

public sealed class AiShotBoundaryPayloadTests
{
    [Fact]
    public void Read_ParsesContractVersion2()
    {
        var read = AiShotBoundaryPayload.Read(Node());

        Assert.Equal(ShotBoundaryReadError.None, read.Error);
        var node = Assert.IsType<ShotBoundaryNode>(read.Node);
        Assert.Equal(2, node.SchemaVersion);
        Assert.Equal("omnishotcut_shot_boundaries", node.Model);
        Assert.Equal("1.0", node.ModelVersion);
        Assert.Equal("clean_shot", node.Mode);
        Assert.Equal(30.0, node.Fps);
        Assert.Equal(4.0, node.DurationSeconds);
        Assert.Equal(120, node.SourceFrameCount);
        Assert.Equal("ffmpeg_cpu", node.DecodeBackend);
        Assert.Equal(0.0, node.DurationMismatchSeconds);
        Assert.Equal(
            [
                new ShotBoundaryElement(0.0, 2.0, 0, 60, "General", null, false),
                new ShotBoundaryElement(2.0, 2.5, 60, 75, null, null, true),
                new ShotBoundaryElement(2.5, 4.0, 75, 120, "General", "Hard_Cut", false),
            ],
            node.Boundaries);
        Assert.NotNull(node.LabelCounts);
    }

    [Theory]
    [InlineData(null)]
    [InlineData(1)]
    public void Read_RefusesContractVersion1(int? schemaVersion)
    {
        var json = JsonNode.Parse(Node())!.AsObject();
        json.Remove("schema_version");
        if (schemaVersion is not null)
        {
            json["schema_version"] = schemaVersion;
        }

        var read = AiShotBoundaryPayload.Read(json.ToJsonString());

        Assert.Null(read.Node);
        Assert.Equal(ShotBoundaryReadError.UnsupportedSchema, read.Error);
        Assert.Contains("version 1", read.Message);
    }

    [Theory]
    [InlineData(null, "Missing")]
    [InlineData("", "Missing")]
    [InlineData("not json", "Malformed")]
    [InlineData("[]", "Malformed")]
    [InlineData("""{"schema_version": 2, "duration_seconds": 4.0}""", "Malformed")]
    [InlineData("""{"schema_version": 2, "duration_seconds": 0, "boundaries": [{"start_seconds": 0, "end_seconds": 1}]}""", "Malformed")]
    [InlineData("""{"schema_version": 2, "duration_seconds": 4.0, "boundaries": []}""", "Malformed")]
    [InlineData("""{"schema_version": 2, "duration_seconds": 4.0, "boundaries": [{"start_seconds": 0}]}""", "Malformed")]
    public void Read_ReportsMissingAndMalformedOutput(string? json, string expected)
    {
        var read = AiShotBoundaryPayload.Read(json);

        Assert.Null(read.Node);
        Assert.Equal(Enum.Parse<ShotBoundaryReadError>(expected), read.Error);
        Assert.False(string.IsNullOrWhiteSpace(read.Message));
    }

    internal static string Node(Action<JsonObject>? edit = null)
    {
        var node = JsonNode.Parse("""
        {
          "schema_version": 2,
          "model": "omnishotcut_shot_boundaries",
          "model_version": "1.0",
          "mode": "clean_shot",
          "fps": 30.0,
          "duration_seconds": 4.0,
          "source_frame_count": 120,
          "duration_mismatch_seconds": 0.0,
          "window_frames": 100,
          "context_frames": 0,
          "decode_backend": "ffmpeg_cpu",
          "decode_wait_seconds": 0.1,
          "inference_seconds": 0.2,
          "analyze_seconds": 0.3,
          "boundaries": [
            { "start_frame": 0, "end_frame": 60, "start_seconds": 0.0, "end_seconds": 2.0, "shot_type": "General", "transition_in": null, "filled": false },
            { "start_frame": 60, "end_frame": 75, "start_seconds": 2.0, "end_seconds": 2.5, "shot_type": null, "transition_in": null, "filled": true },
            { "start_frame": 75, "end_frame": 120, "start_seconds": 2.5, "end_seconds": 4.0, "shot_type": "General", "transition_in": "Hard_Cut", "filled": false }
          ],
          "label_counts": { "shot_type": { "General": 2 }, "transition_in": { "Hard_Cut": 1 } },
          "shots": [
            { "start_frame": 0, "end_frame": 60, "intra": "General", "inter": "New_Start" },
            { "start_frame": 75, "end_frame": 120, "intra": "General", "inter": "Hard_Cut" }
          ]
        }
        """)!.AsObject();
        edit?.Invoke(node);
        return node.ToJsonString();
    }
}

public sealed class AiShotsContributorTests
{
    private const string RunId = "0123456789abcdef0123456789abcdef";

    [Fact]
    public void Describe_AdvertisesOneWholeVideoShotBoundaryClaimThatIsNeverTickedForTheUser()
    {
        var (contributor, _, _) = Create();

        var descriptor = contributor.Describe();

        Assert.Equal("cove.community.ai.shots", descriptor.ExtensionId);
        var claim = Assert.Single(descriptor.Claims);
        Assert.Equal(AiShotsContributor.ClaimId, claim.ClaimId);
        Assert.Equal(AiMediaKinds.Video, claim.MediaKind);
        Assert.Equal("temporal_segmentation", claim.WantCapability);
        Assert.Equal("asset", claim.WantScope);
        Assert.Equal("shots", claim.CapabilityId);
        Assert.Equal("detector", claim.ModelBindingSlotId);
        var feature = Assert.Single(descriptor.Capabilities);
        Assert.False(feature.SelectedByDefault);
        var slot = Assert.Single(feature.ModelBindingSlots!);
        Assert.Equal(["temporal_segmentation"], slot.RequiredCapabilities);
        Assert.Equal(["asset"], slot.RequiredScopes);
        Assert.Equal(["shot_boundaries"], slot.RequiredCategories);
        Assert.Contains("Force rerun replaces them", feature.Description);
    }

    [Fact]
    public async Task Plan_RunsWhenTheFileHasNoShots()
    {
        var (contributor, shots, _) = Create();

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target()));

        Assert.Equal(AiClaimPlanningVerdict.Run, decision.Verdict);
        Assert.False(decision.ReplacesExistingResults);
        Assert.Equal([5], Assert.Single(shots.SummaryRequests));
    }

    [Fact]
    public async Task Plan_SkipsWhenTheFileAlreadyHasShots()
    {
        var (contributor, shots, _) = Create();
        shots.Existing[5] = new VideoShotSetDto { Id = 1, FileId = 5, ShotCount = 12 };

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target()));

        Assert.Equal(AiClaimPlanningVerdict.Skip, decision.Verdict);
        Assert.Contains("already has 12 shots", decision.Reason);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Plan_ForcedReplacesExistingShots(bool claimsExplicitlySelected)
    {
        var (contributor, shots, _) = Create();
        shots.Existing[5] = new VideoShotSetDto { Id = 1, FileId = 5, ShotCount = 12 };

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target() with
        {
            ForcedClaimIds = [AiShotsContributor.ClaimId],
            ClaimsExplicitlySelected = claimsExplicitlySelected,
        }));

        Assert.Equal(AiClaimPlanningVerdict.Run, decision.Verdict);
        Assert.True(decision.ReplacesExistingResults);
        Assert.Contains("12 shots will be replaced", decision.Reason);
    }

    [Fact]
    public async Task Plan_ForcedWithNothingStoredIsAFirstRun()
    {
        var (contributor, _, _) = Create();

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target() with { ForcedClaimIds = [AiShotsContributor.ClaimId] }));

        Assert.Equal(AiClaimPlanningVerdict.Run, decision.Verdict);
        Assert.False(decision.ReplacesExistingResults);
    }

    [Fact]
    public async Task Plan_SkipsRunsThatDidNotSelectShotBoundariesWithoutLookingUpTheFile()
    {
        var (contributor, shots, files) = Create();

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target() with { ClaimsExplicitlySelected = false }));

        Assert.Equal(AiClaimPlanningVerdict.Skip, decision.Verdict);
        Assert.Equal(0, files.Calls);
        Assert.Empty(shots.SummaryRequests);
    }

    [Fact]
    public async Task Plan_SkipsWhenResultsAreNotStored()
    {
        var (contributor, _, files) = Create();

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target() with
        {
            DispatchResults = false,
            ForcedClaimIds = [AiShotsContributor.ClaimId],
        }));

        Assert.Equal(AiClaimPlanningVerdict.Skip, decision.Verdict);
        Assert.Equal(0, files.Calls);
    }

    [Theory]
    [InlineData("image", 42)]
    [InlineData("video", null)]
    [InlineData(null, 42)]
    public async Task Plan_SkipsTargetsThatAreNotCoveVideos(string? entityType, int? entityId)
    {
        var (contributor, _, files) = Create();

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target() with
        {
            HostEntityType = entityType,
            HostEntityId = entityId,
            ForcedClaimIds = [AiShotsContributor.ClaimId],
        }));

        Assert.Equal(AiClaimPlanningVerdict.Skip, decision.Verdict);
        Assert.Contains("Cove videos only", decision.Reason);
        Assert.Equal(0, files.Calls);
    }

    [Fact]
    public async Task Plan_ResolvesTheFileFromTheAnalysedPath()
    {
        var (contributor, shots, files) = Create();
        files.Files[42] = [File(10, "E:/media/other.mp4"), File(11, "E:/media/example.mp4")];

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target() with { HostFileId = null }));

        Assert.Equal(AiClaimPlanningVerdict.Run, decision.Verdict);
        Assert.Equal([11], Assert.Single(shots.SummaryRequests));
    }

    [Fact]
    public async Task Plan_DoesNotTrustANamedFileOfAnotherVideo()
    {
        var (contributor, shots, files) = Create();
        files.Files[42] = [File(10, "E:/media/other.mp4"), File(11, "E:/media/example.mp4")];

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target() with { HostFileId = 999 }));

        Assert.Equal(AiClaimPlanningVerdict.Run, decision.Verdict);
        Assert.Equal([11], Assert.Single(shots.SummaryRequests));
    }

    [Fact]
    public async Task Plan_SkipsWhenTheFileCannotBeIdentified()
    {
        var (contributor, shots, files) = Create();
        files.Files[42] = [File(10, "E:/media/other.mp4"), File(11, "E:/media/third.mp4")];

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target() with { HostFileId = null }));

        Assert.Equal(AiClaimPlanningVerdict.Skip, decision.Verdict);
        Assert.Empty(shots.SummaryRequests);
    }

    [Fact]
    public async Task Plan_ASingleFileMustStillBeTheAnalysedOne()
    {
        var (contributor, shots, files) = Create();
        files.Files[42] = [File(10, "E:/media/other.mp4")];

        var decision = Assert.Single(await contributor.PlanClaimsAsync(Target() with { HostFileId = null }));

        Assert.Equal(AiClaimPlanningVerdict.Skip, decision.Verdict);
        Assert.Empty(shots.SummaryRequests);
    }

    [Fact]
    public async Task Plan_AnswersOnlyForItsOwnClaim()
    {
        var (contributor, _, _) = Create();
        var other = new AiCapabilityClaim("other.claim", "Other", AiMediaKinds.Video, "tagging", "frame", "tags");

        var decisions = await contributor.PlanClaimsAsync(Target() with { Claims = [other] });

        Assert.Empty(decisions);
    }

    [Fact]
    public async Task Dispatch_WritesEveryFieldAndSkipsExistingSetsByDefault()
    {
        var (contributor, shots, _) = Create();

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5)));

        var write = Assert.Single(shots.Writes);
        Assert.Equal(5, write.FileId);
        Assert.Equal(VideoShotWriteMode.SkipIfExists, write.WriteMode);
        Assert.Null(write.Expected);
        Assert.Null(write.EditedAt);
        Assert.Equal("ext:ai.shots", write.SourceKey);
        Assert.Equal(RunId, write.SourceRunId);
        Assert.Equal("omnishotcut_shot_boundaries", write.Model);
        Assert.Equal("1.0", write.ModelVersion);
        Assert.Equal("clean_shot", write.Mode);
        Assert.Equal("ffmpeg_cpu", write.DecodeBackend);
        Assert.Equal(30.0, write.Fps);
        Assert.Equal(120, write.FrameCount);
        Assert.Equal(4.0, write.DurationSec);
        Assert.Equal(
            [
                new VideoShotInput { StartSec = 0.0, EndSec = 2.0, StartFrame = 0, EndFrame = 60, ShotType = "General" },
                new VideoShotInput { StartSec = 2.0, EndSec = 2.5, StartFrame = 60, EndFrame = 75 },
                new VideoShotInput { StartSec = 2.5, EndSec = 4.0, StartFrame = 75, EndFrame = 120, ShotType = "General", TransitionIn = "Hard_Cut" },
            ],
            write.Shots);
        var payload = write.Payload!.Value;
        Assert.Equal(2, payload.GetProperty("schemaVersion").GetInt32());
        Assert.Equal(100, payload.GetProperty("windowFrames").GetInt32());
        Assert.Equal(1, payload.GetProperty("filledShots").GetInt32());
        Assert.Equal(2, payload.GetProperty("labelCounts").GetProperty("shot_type").GetProperty("General").GetInt32());
        Assert.Equal(new Dictionary<string, int> { ["shots"] = 3 }, result.PreparedCounts);
        Assert.Null(result.Notes);
    }

    [Fact]
    public async Task Dispatch_FilledShotsCarryNoLabels()
    {
        var (contributor, shots, _) = Create();
        var node = AiShotBoundaryPayloadTests.Node(json =>
        {
            var filled = json["boundaries"]![1]!.AsObject();
            filled["shot_type"] = "Dissolve";
            filled["transition_in"] = "Hard_Cut";
        });

        await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node));

        var filledShot = Assert.Single(shots.Writes).Shots[1];
        Assert.Null(filledShot.ShotType);
        Assert.Null(filledShot.TransitionIn);
    }

    [Fact]
    public async Task Dispatch_TruncatesLabelsToCovesLimitWithoutSplittingACharacter()
    {
        var (contributor, shots, _) = Create();
        var longLabel = new string('a', 150);
        var emojiAtTheLimit = new string('b', 99) + "\U0001F3AC" + "tail";
        var node = AiShotBoundaryPayloadTests.Node(json =>
        {
            json["boundaries"]![0]!["shot_type"] = longLabel;
            json["boundaries"]![2]!["transition_in"] = emojiAtTheLimit;
        });

        await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node));

        var write = Assert.Single(shots.Writes);
        Assert.Equal(new string('a', 100), write.Shots[0].ShotType);
        Assert.Equal(new string('b', 99), write.Shots[2].TransitionIn);
    }

    [Theory]
    [InlineData(null)]
    [InlineData(0.0)]
    public async Task Dispatch_DropsFramesWithoutAFrameRate(double? fps)
    {
        var (contributor, shots, _) = Create();
        var node = AiShotBoundaryPayloadTests.Node(json =>
        {
            json.Remove("fps");
            if (fps is not null)
            {
                json["fps"] = fps;
            }
        });

        await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node));

        var write = Assert.Single(shots.Writes);
        Assert.Null(write.Fps);
        Assert.Null(write.FrameCount);
        Assert.All(write.Shots, static shot => Assert.True(shot.StartFrame is null && shot.EndFrame is null));
    }

    [Fact]
    public async Task Dispatch_LeavesOutLabelCountsThatWouldExceedCovesPayloadLimit()
    {
        var (contributor, shots, _) = Create();
        var node = AiShotBoundaryPayloadTests.Node(json =>
        {
            var counts = new JsonObject();
            for (var index = 0; index < 5000; index++)
            {
                counts[$"label_{index:D5}_{new string('x', 20)}"] = index;
            }

            json["label_counts"] = new JsonObject { ["shot_type"] = counts };
        });

        await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node));

        var payload = Assert.Single(shots.Writes).Payload!.Value;
        Assert.False(payload.TryGetProperty("labelCounts", out _));
        Assert.Equal(2, payload.GetProperty("schemaVersion").GetInt32());
        Assert.True(Encoding.UTF8.GetByteCount(payload.GetRawText()) <= Cove.Core.Common.VideoShotRules.MaxPayloadBytes);
    }

    [Fact]
    public async Task Dispatch_LeavesOutLabelCountsCarryingANulCharacter()
    {
        var (contributor, shots, _) = Create();
        var node = AiShotBoundaryPayloadTests.Node(json =>
            json["label_counts"] = new JsonObject { ["shot_type"] = new JsonObject { ["Gen\0eral"] = 2 } });

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node));

        var payload = Assert.Single(shots.Writes).Payload!.Value;
        Assert.False(payload.TryGetProperty("labelCounts", out _));
        Assert.Equal(new Dictionary<string, int> { ["shots"] = 3 }, result.PreparedCounts);
    }

    [Fact]
    public async Task Dispatch_StripsNulCharactersFromLabelsAndModelStrings()
    {
        var (contributor, shots, _) = Create();
        var node = AiShotBoundaryPayloadTests.Node(json =>
        {
            json["model"] = "omni\0shot";
            json["model_version"] = "1\0.0";
            json["mode"] = "clean\0_shot";
            json["decode_backend"] = "ffmpeg\0_cpu";
            json["boundaries"]![0]!["shot_type"] = "Gen\0eral";
            json["boundaries"]![2]!["transition_in"] = "Hard\0_Cut";
        });

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node));

        var write = Assert.Single(shots.Writes);
        Assert.Equal("omnishot", write.Model);
        Assert.Equal("1.0", write.ModelVersion);
        Assert.Equal("clean_shot", write.Mode);
        Assert.Equal("ffmpeg_cpu", write.DecodeBackend);
        Assert.Equal("General", write.Shots[0].ShotType);
        Assert.Equal("Hard_Cut", write.Shots[2].TransitionIn);
        Assert.Equal(new Dictionary<string, int> { ["shots"] = 3 }, result.PreparedCounts);
    }

    [Theory]
    [InlineData("abc", 5, "abc")]
    [InlineData("a\0b\0c", 5, "abc")]
    [InlineData("abcdef", 4, "abcd")]
    [InlineData("abc\U0001F3ACd", 4, "abc")]
    [InlineData("\0abc\U0001F3AC", 5, "abc\U0001F3AC")]
    public void Clean_DropsNulAndTruncatesWithoutSplittingACharacter(string value, int maxLength, string expected)
        => Assert.Equal(expected, AiShotsContributor.Clean(value, maxLength));

    [Fact]
    public async Task Dispatch_ForcedRunReplacesAndReportsReplacedEdits()
    {
        var (contributor, shots, _) = Create();
        shots.NextResult = new VideoShotWriteResult
        {
            Outcome = VideoShotWriteOutcome.Replaced,
            ReplacedEditedSet = true,
            Set = new VideoShotSetDto { Id = 1, FileId = 5, ShotCount = 3 },
        };

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5, forced: true)));

        Assert.Equal(VideoShotWriteMode.Replace, Assert.Single(shots.Writes).WriteMode);
        Assert.Equal(new Dictionary<string, int> { ["shots"] = 3 }, result.PreparedCounts);
        Assert.Contains("edited by hand", Assert.Single(result.Notes!));
    }

    [Fact]
    public async Task Dispatch_ReportsAKeptSet()
    {
        var (contributor, shots, _) = Create();
        shots.NextResult = new VideoShotWriteResult { Outcome = VideoShotWriteOutcome.SkippedExisting };

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5)));

        Assert.Equal(new Dictionary<string, int> { ["shotSetsSkipped"] = 1 }, result.PreparedCounts);
        Assert.Contains("kept", Assert.Single(result.Notes!));
    }

    [Theory]
    [InlineData(VideoShotWriteOutcome.Invalid)]
    [InlineData(VideoShotWriteOutcome.Conflict)]
    [InlineData(VideoShotWriteOutcome.FileNotFound)]
    public async Task Dispatch_ReportsARejectedWrite(VideoShotWriteOutcome outcome)
    {
        var (contributor, shots, _) = Create();
        shots.NextResult = new VideoShotWriteResult { Outcome = outcome, Reason = "the partition has a gap" };

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5)));

        Assert.Null(result.PreparedCounts);
        var note = Assert.Single(result.Notes!);
        Assert.Contains(outcome.ToString(), note);
        Assert.Contains("the partition has a gap", note);
    }

    [Theory]
    [InlineData(1.5, "longer than the 1.5s Cove has for the file")]
    [InlineData(10.0, "less than half of the 10s Cove has for the file")]
    public async Task Dispatch_ExplainsADurationTheFileCannotHave(double fileSeconds, string explanation)
    {
        // The analysed node covers 4 seconds.
        var (contributor, shots, files) = Create();
        files.Files[42] = [File(5, "E:/media/example.mp4", fileSeconds)];
        shots.NextResult = new VideoShotWriteResult { Outcome = VideoShotWriteOutcome.Invalid, Reason = "Cove's own reason" };

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5)));

        var note = Assert.Single(result.Notes!);
        Assert.Contains("the analysis covered", note);
        Assert.Contains(explanation, note);
        Assert.Contains("Rescan the file", note);
    }

    [Theory]
    [InlineData(4.0, 4.0)]
    [InlineData(4.0, 5.9)]
    [InlineData(4.0, 8.0)]
    [InlineData(200.0, 202.0)]
    public void DescribeDurationProblem_AcceptsWhatCoveAccepts(double analysedSeconds, double fileSeconds)
        => Assert.Null(AiShotsContributor.DescribeDurationProblem(analysedSeconds, fileSeconds));

    [Theory]
    [InlineData(6.1, 4.0)]
    [InlineData(203.1, 200.0)]
    [InlineData(3.9, 8.0)]
    public void DescribeDurationProblem_ExplainsWhatCoveRefuses(double analysedSeconds, double fileSeconds)
        => Assert.NotNull(AiShotsContributor.DescribeDurationProblem(analysedSeconds, fileSeconds));

    [Fact]
    public async Task Dispatch_TakesTheModelVersionFromTheResultWhenTheNodeHasNone()
    {
        var (contributor, shots, _) = Create();
        var node = AiShotBoundaryPayloadTests.Node(json => json.Remove("model_version"));

        await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node, models:
        [
            new AiModelDescriptor("omnishotcut_shot_boundaries", "omnishotcut_shot_boundaries", Version: "3.1"),
        ]));

        Assert.Equal("3.1", Assert.Single(shots.Writes).ModelVersion);
    }

    [Fact]
    public async Task Dispatch_DropsFramesUnlessEveryShotHasThem()
    {
        var (contributor, shots, _) = Create();
        var node = AiShotBoundaryPayloadTests.Node(json => json["boundaries"]![1]!.AsObject().Remove("end_frame"));

        await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node));

        var write = Assert.Single(shots.Writes);
        Assert.Null(write.FrameCount);
        Assert.All(write.Shots, static shot => Assert.True(shot.StartFrame is null && shot.EndFrame is null));
    }

    [Fact]
    public async Task Dispatch_WithoutOutputNotesItAndWritesNothing()
    {
        var (contributor, shots, _) = Create();

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node: null));

        Assert.Empty(shots.Writes);
        Assert.Contains("no shot-boundary output", Assert.Single(result.Notes!));
    }

    [Fact]
    public async Task Dispatch_RefusesContractVersion1()
    {
        var (contributor, shots, _) = Create();
        var node = AiShotBoundaryPayloadTests.Node(json => json.Remove("schema_version"));

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5), node));

        Assert.Empty(shots.Writes);
        Assert.Contains("update it", Assert.Single(result.Notes!));
    }

    [Fact]
    public async Task Dispatch_DoesNotThrowWhenStorageFails()
    {
        var (contributor, shots, _) = Create();
        shots.ThrowOnWrite = new InvalidOperationException("database unavailable");

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5)));

        Assert.Contains("database unavailable", Assert.Single(result.Notes!));
    }

    [Fact]
    public async Task Dispatch_ResolvesTheFileFromTheAnalysedPathWhenAiCoreDidNotSay()
    {
        var (contributor, shots, files) = Create();
        files.Files[42] = [File(10, "E:/media/other.mp4"), File(11, "E:\\media\\example.mp4")];

        await contributor.DispatchAsync(Request(metadata: null));

        Assert.Equal(11, Assert.Single(shots.Writes).FileId);
    }

    [Fact]
    public async Task Dispatch_DoesNotTrustANamedFileOfAnotherVideo()
    {
        var (contributor, shots, files) = Create();
        files.Files[42] = [File(10, "E:/media/other.mp4"), File(11, "E:/media/example.mp4")];

        await contributor.DispatchAsync(Request(Metadata(hostFileId: 999)));

        Assert.Equal(11, Assert.Single(shots.Writes).FileId);
    }

    [Fact]
    public async Task Dispatch_ASingleFileMustStillBeTheAnalysedOne()
    {
        var (contributor, shots, files) = Create();
        files.Files[42] = [File(10, "E:/media/other.mp4")];

        var result = await contributor.DispatchAsync(Request(metadata: null));

        Assert.Empty(shots.Writes);
        Assert.Contains("could not be identified", Assert.Single(result.Notes!));
    }

    [Fact]
    public async Task Dispatch_WritesNothingWhenTheFileIsAmbiguous()
    {
        var (contributor, shots, files) = Create();
        files.Files[42] = [File(10, "E:/media/other.mp4"), File(11, "E:/media/third.mp4")];

        var result = await contributor.DispatchAsync(Request(metadata: null));

        Assert.Empty(shots.Writes);
        Assert.Contains("could not be identified", Assert.Single(result.Notes!));
    }

    [Theory]
    [InlineData("image", 42)]
    [InlineData("video", null)]
    public async Task Dispatch_WritesNothingForARunWithoutACoveVideo(string entityType, int? entityId)
    {
        var (contributor, shots, _) = Create();
        var request = Request(Metadata(hostFileId: 5));

        var result = await contributor.DispatchAsync(request with
        {
            Context = request.Context with { HostEntityType = entityType, HostEntityId = entityId },
        });

        Assert.Empty(shots.Writes);
        Assert.Contains("no Cove video", Assert.Single(result.Notes!));
    }

    [Fact]
    public async Task Dispatch_IgnoresRequestsWithoutItsClaim()
    {
        var (contributor, shots, _) = Create();
        var other = new AiCapabilityClaim("other.claim", "Other", AiMediaKinds.Video, "tagging", "frame", "tags");

        var result = await contributor.DispatchAsync(Request(Metadata(hostFileId: 5)) with { Claims = [other] });

        Assert.Equal(0, result.ClaimCount);
        Assert.Empty(shots.Writes);
    }

    [Fact]
    public async Task Extension_PublishesItsContributorUnderItsManifestId()
    {
        var extension = new AiShotsExtension();
        var manifest = JsonSerializer.Deserialize<ExtensionManifestFile>(
            System.IO.File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "extension.json")),
            new JsonSerializerOptions(JsonSerializerDefaults.Web))!;
        ((IManifestAware)extension).ApplyManifest(manifest);

        var services = new ServiceCollection();
        services.AddLogging();
        var exchange = new ExtensionServiceExchange();
        services.AddSingleton<IExtensionServiceExchange>(exchange);
        extension.ConfigureServices(services, new ExtensionContext
        {
            Configuration = new ConfigurationBuilder().Build(),
            DataDirectory = Path.GetTempPath(),
            CoveVersion = "1.5.2",
        });
        await using var provider = services.BuildServiceProvider();
        await extension.InitializeAsync(provider);

        var contributor = Assert.Single(exchange.GetAll<IAiCapabilityContributor>());
        Assert.IsType<AiShotsContributor>(contributor);
        Assert.Equal(manifest.Id, contributor.Describe().ExtensionId);
        Assert.Contains(manifest.TutorialTopics ?? [], static topic => topic.Id == "cove.community.ai.shots" && topic.ParentTopicId == "cove.ai");
    }

    private static (AiShotsContributor Contributor, FakeVideoShotService Shots, FakeFileLookup Files) Create()
    {
        var shots = new FakeVideoShotService();
        var files = new FakeFileLookup();
        files.Files[42] = [File(5, "E:/media/example.mp4")];
        var services = new ServiceCollection();
        services.AddScoped<IVideoShotService>(_ => shots);
        services.AddScoped<IAiShotsFileLookup>(_ => files);
        var provider = services.BuildServiceProvider();
        var persistence = new AiShotsPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
        return (new AiShotsContributor(persistence, NullLogger<AiShotsContributor>.Instance), shots, files);
    }

    private static VideoFileInfo File(int fileId, string path, double duration = 4.0) => new(fileId, path, duration);

    private static AiClaimPlanningTarget Target()
        => new()
        {
            RunId = RunId,
            MediaKind = AiMediaKinds.Video,
            Subject = "E:/media/example.mp4",
            HostEntityType = "video",
            HostEntityId = 42,
            HostFileId = 5,
            Claims = [ShotsClaim()],
            ClaimsExplicitlySelected = true,
            DispatchResults = true,
        };

    private static AiCapabilityClaim ShotsClaim()
        => Assert.Single(new AiShotsContributor(null!, NullLogger<AiShotsContributor>.Instance).Describe().Claims);

    private static Dictionary<string, string> Metadata(int? hostFileId, bool forced = false)
    {
        var metadata = new Dictionary<string, string>
        {
            ["source"] = "cove.community.ai.core",
            ["extensionId"] = "cove.community.ai.shots",
            [AiDispatchMetadataKeys.ForcedClaimIds] = forced ? AiShotsContributor.ClaimId : string.Empty,
        };
        if (hostFileId is int fileId)
        {
            metadata[AiDispatchMetadataKeys.HostFileId] = fileId.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        return metadata;
    }

    private static AiDispatchRequest Request(
        IReadOnlyDictionary<string, string>? metadata,
        string? node = "",
        IReadOnlyList<AiModelDescriptor>? models = null)
    {
        var other = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var json = node == "" ? AiShotBoundaryPayloadTests.Node() : node;
        if (json is not null)
        {
            other[AiShotBoundaryPayload.OutputKey] = json;
        }

        return new AiDispatchRequest(
            new AiRunContext(RunId, AiMediaKinds.Video, "E:/media/example.mp4", "E:/media/example.mp4", "video", 42),
            [ShotsClaim()],
            new AiAnalyzeResult
            {
                MediaKind = AiMediaKinds.Video,
                Models = models ?? [],
                AssetAnalysis = new AiAnalysisNode { Other = other },
            },
            metadata);
    }

    private sealed class FakeVideoShotService : IVideoShotService
    {
        public Dictionary<int, VideoShotSetDto> Existing { get; } = [];

        public List<int[]> SummaryRequests { get; } = [];

        public List<VideoShotSetWrite> Writes { get; } = [];

        public VideoShotWriteResult? NextResult { get; set; }

        public Exception? ThrowOnWrite { get; set; }

        // Like Cove, summaries carry neither shots nor the payload.
        public Task<IReadOnlyDictionary<int, VideoShotSetDto>> GetSummariesForFilesAsync(IReadOnlyCollection<int> fileIds, CancellationToken cancellationToken = default)
        {
            SummaryRequests.Add(fileIds.ToArray());
            return Task.FromResult<IReadOnlyDictionary<int, VideoShotSetDto>>(
                fileIds.Where(Existing.ContainsKey).ToDictionary(static id => id, id => Existing[id] with { Payload = null, Shots = null }));
        }

        // Like Cove, a payload over the size limit or carrying U+0000 is refused, a written set's summary carries no
        // payload, and EditedAt is kept in UTC.
        public Task<VideoShotWriteResult> WriteSetAsync(VideoShotSetWrite write, CancellationToken cancellationToken = default)
        {
            if (ThrowOnWrite is not null)
            {
                throw ThrowOnWrite;
            }

            Writes.Add(write);
            if (write.Payload is { ValueKind: not (JsonValueKind.Undefined or JsonValueKind.Null) } payload)
            {
                if (Encoding.UTF8.GetByteCount(payload.GetRawText()) > Cove.Core.Common.VideoShotRules.MaxPayloadBytes)
                {
                    return Task.FromResult(new VideoShotWriteResult { Outcome = VideoShotWriteOutcome.Invalid, Reason = "The payload is too large." });
                }

                if (payload.GetRawText().Contains("\\u0000", StringComparison.Ordinal))
                {
                    return Task.FromResult(new VideoShotWriteResult { Outcome = VideoShotWriteOutcome.Invalid, Reason = "The payload may not contain the character U+0000." });
                }
            }

            return Task.FromResult(NextResult ?? new VideoShotWriteResult
            {
                Outcome = VideoShotWriteOutcome.Written,
                Set = new VideoShotSetDto
                {
                    Id = 1,
                    FileId = write.FileId,
                    ShotCount = write.Shots.Count,
                    EditedAt = write.EditedAt?.ToUniversalTime(),
                },
            });
        }

        public Task<VideoShotSetDto?> GetForFileAsync(int fileId, CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task<VideoShotSetDto?> GetForVideoAsync(int videoId, CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task<IReadOnlyList<VideoShotSetDto>> ListForVideoAsync(int videoId, CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task<VideoShotEditResult> SplitAsync(VideoShotSplitRequest request, CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task<VideoShotEditResult> MergeAsync(VideoShotMergeRequest request, CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task<VideoShotEditResult> MoveCutAsync(VideoShotMoveCutRequest request, CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task<VideoShotEditResult> ReplaceShotsAsync(VideoShotReplaceRequest request, CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task<VideoShotEditResult> DeleteAsync(VideoShotDeleteRequest request, CancellationToken cancellationToken = default) => throw new NotSupportedException();
    }

    private sealed class FakeFileLookup : IAiShotsFileLookup
    {
        public Dictionary<int, IReadOnlyList<VideoFileInfo>> Files { get; } = [];

        public int Calls { get; private set; }

        public Task<IReadOnlyList<VideoFileInfo>> GetVideoFilesAsync(int videoId, CancellationToken ct = default)
        {
            Calls++;
            return Task.FromResult(Files.TryGetValue(videoId, out var files) ? files : []);
        }
    }
}

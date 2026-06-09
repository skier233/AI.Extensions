using System.Text.Json;

using AI.Core;
using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiRunQueueServiceTests
{
    [Fact]
    public async Task ExecuteAsync_DisablesAiServerRequestTimeoutForQueuedWork()
    {
        var orchestrator = new RecordingOrchestrator();
        var service = new AiRunQueueService(
            orchestrator,
            new FixedTargetResolver([
                new AiResolvedRunTarget(
                    UnitId: "video:23",
                    Label: "Video 23",
                    Path: "E:/test/Content/Videos/example.mp4",
                    EntityId: 23,
                    EntityType: "video")
            ]),
            new StubJobService(),
                    CreateScopeFactory(orchestrator),
            NullLogger<AiRunQueueService>.Instance);

        await service.ExecuteAsync(
            new AiCoreConnectionSettings
            {
                RequestTimeoutSeconds = 900,
            }.Normalize(),
            new AiQueueRunRequest
            {
                MediaKind = AiMediaKinds.Video,
                Paths = ["E:/test/Content/Videos/example.mp4"],
                ClaimIds = ["faces.video.detection", "faces.video.embedding"],
            },
            new NullJobProgress());

        Assert.Equal(0, orchestrator.LastSettings?.RequestTimeoutSeconds);
    }

    [Fact]
    public async Task QueueAsync_ExecutesQueuedWorkInFreshScope()
    {
        var requestOrchestrator = new RecordingOrchestrator(throwWhenDisposed: true);
        var queuedOrchestrator = new RecordingOrchestrator();
        var jobService = new CapturingJobService();
        var service = new AiRunQueueService(
            requestOrchestrator,
            new FixedTargetResolver([
                new AiResolvedRunTarget(
                    UnitId: "video:23",
                    Label: "Video 23",
                    Path: "E:/test/Content/Videos/example.mp4",
                    EntityId: 23,
                    EntityType: "video")
            ]),
            jobService,
            CreateScopeFactory(queuedOrchestrator),
            NullLogger<AiRunQueueService>.Instance);

        var response = await service.QueueAsync(
            new AiCoreConnectionSettings
            {
                RequestTimeoutSeconds = 900,
            }.Normalize(),
            new AiQueueRunRequest
            {
                MediaKind = AiMediaKinds.Video,
                Paths = ["E:/test/Content/Videos/example.mp4"],
                ClaimIds = ["faces.video.detection", "faces.video.embedding"],
            });

        requestOrchestrator.DisposeScope();
        await jobService.RunQueuedAsync();

        Assert.Equal("job-1", response.JobId);
        Assert.Equal(0, requestOrchestrator.RunVideoCallCount);
        Assert.Equal(1, queuedOrchestrator.RunVideoCallCount);
        Assert.Equal(0, queuedOrchestrator.LastSettings?.RequestTimeoutSeconds);
    }

    [Fact]
    public async Task ExecuteAsync_RegistersAllTargetsAndReportsIndexedVideoProgress()
    {
        var orchestrator = new RecordingOrchestrator();
        var progress = new RecordingJobProgress();
        var service = new AiRunQueueService(
            orchestrator,
            new FixedTargetResolver([
                new AiResolvedRunTarget(
                    UnitId: "video:23",
                    Label: "Video 23",
                    Path: "E:/test/Content/Videos/example-23.mp4",
                    EntityId: 23,
                    EntityType: "video"),
                new AiResolvedRunTarget(
                    UnitId: "video:24",
                    Label: "Video 24",
                    Path: "E:/test/Content/Videos/example-24.mp4",
                    EntityId: 24,
                    EntityType: "video")
            ]),
            new StubJobService(),
            CreateScopeFactory(orchestrator),
            NullLogger<AiRunQueueService>.Instance);

        await service.ExecuteAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiQueueRunRequest
            {
                EntityType = "video",
                MediaKind = AiMediaKinds.Video,
                EntityIds = [23, 24],
                ClaimIds = ["faces.video.detection"],
            },
            progress);

        Assert.Equal(["video:23", "video:24"], progress.FirstStartedUnitIds(2));
        Assert.Contains(progress.UnitMessages, static message => message == "Processing Video 1 of 2: Video 23");
        Assert.Contains(progress.UnitMessages, static message => message == "Processing Video 2 of 2: Video 24");
    }

    [Fact]
    public async Task ResolveAsync_RequestsAllSelectedVideoIds()
    {
        var videoRepository = new PagingVideoRepository(
            Enumerable.Range(1, 60).Select(static id => new Video
            {
                Id = id,
                Title = $"Video {id}",
                Files = [new VideoFile { Path = $"E:/test/{id}.mp4", Duration = id }],
            }).ToList());
        var resolver = new AiRunTargetResolver(videoRepository, new ThrowingImageRepository());

        var targets = await resolver.ResolveAsync(new AiQueueRunRequest
        {
            EntityType = "video",
            MediaKind = AiMediaKinds.Video,
            EntityIds = Enumerable.Range(1, 60).ToList(),
        });

        Assert.Equal(60, targets.Count);
        Assert.Equal(60, videoRepository.LastFindFilter?.PerPage);
    }

    private static IServiceScopeFactory CreateScopeFactory(IAiCoreOrchestrator orchestrator)
        => new ServiceCollection()
            .AddScoped(_ => orchestrator)
            .BuildServiceProvider()
            .GetRequiredService<IServiceScopeFactory>();

    private sealed class RecordingOrchestrator : IAiCoreOrchestrator
    {
        private readonly bool _throwWhenDisposed;

        public RecordingOrchestrator(bool throwWhenDisposed = false)
        {
            _throwWhenDisposed = throwWhenDisposed;
        }

        public AiCoreConnectionSettings? LastSettings { get; private set; }

        public int RunVideoCallCount { get; private set; }

        public bool IsDisposed { get; private set; }

        public IReadOnlyList<AiCapabilityDescriptor> GetCapabilities() => [];

        public Task<AiRunResponse> RunImagesAsync(AiCoreConnectionSettings settings, AiRunImagesRequest request, CancellationToken ct = default)
            => Task.FromResult(new AiRunResponse("images", AiMediaKinds.Image, [], EmptyJson(), [], []));

        public Task<AiRunResponse> RunVideoAsync(AiCoreConnectionSettings settings, AiRunVideoRequest request, CancellationToken ct = default)
        {
            if (_throwWhenDisposed && IsDisposed)
                throw new ObjectDisposedException(nameof(RecordingOrchestrator));

            RunVideoCallCount++;
            LastSettings = settings;
            return Task.FromResult(new AiRunResponse("video", AiMediaKinds.Video, [], EmptyJson(), [], []));
        }

        public Task<AiRunResponse> RunAudioAsync(AiCoreConnectionSettings settings, AiRunAudioRequest request, CancellationToken ct = default)
            => Task.FromResult(new AiRunResponse("audio", AiMediaKinds.Audio, [], EmptyJson(), [], []));

        public void DisposeScope() => IsDisposed = true;

        private static JsonElement EmptyJson()
            => JsonDocument.Parse("{}").RootElement.Clone();
    }

    private sealed class FixedTargetResolver(IReadOnlyList<AiResolvedRunTarget> targets) : IAiRunTargetResolver
    {
        public Task<IReadOnlyList<AiResolvedRunTarget>> ResolveAsync(AiQueueRunRequest request, CancellationToken ct = default)
            => Task.FromResult(targets);
    }

    private sealed class StubJobService : IJobService
    {
        public string Enqueue(string type, string description, Func<IJobProgress, CancellationToken, Task> work, bool exclusive = true)
            => throw new NotSupportedException();

        public bool Cancel(string jobId) => false;

        public bool ReorderQueued(string jobId, string? beforeJobId) => false;

        public JobInfo? GetJob(string jobId) => null;

        public IReadOnlyList<JobInfo> GetAllJobs() => [];

        public IReadOnlyList<JobInfo> GetJobHistory() => [];
    }

    private sealed class CapturingJobService : IJobService
    {
        private Func<IJobProgress, CancellationToken, Task>? _work;

        public string Enqueue(string type, string description, Func<IJobProgress, CancellationToken, Task> work, bool exclusive = true)
        {
            _work = work;
            return "job-1";
        }

        public bool Cancel(string jobId) => false;

        public bool ReorderQueued(string jobId, string? beforeJobId) => false;

        public JobInfo? GetJob(string jobId) => null;

        public IReadOnlyList<JobInfo> GetAllJobs() => [];

        public IReadOnlyList<JobInfo> GetJobHistory() => [];

        public Task RunQueuedAsync()
            => (_work ?? throw new InvalidOperationException("No queued work was captured."))(new NullJobProgress(), CancellationToken.None);
    }

    private sealed class NullJobProgress : IJobProgress
    {
        public void Report(double progress, string? subTask = null)
        {
        }
    }

    private sealed class RecordingJobProgress : IJobProgress
    {
        private readonly List<string> _startedUnitIds = [];
        private readonly List<string> _unitMessages = [];

        public IReadOnlyList<string> UnitMessages => _unitMessages;

        public void Report(double progress, string? subTask = null)
        {
        }

        public IJobUnit StartUnit(string unitId, string? label = null)
        {
            _startedUnitIds.Add(unitId);
            return new RecordingJobUnit(_unitMessages);
        }

        public IReadOnlyList<string> FirstStartedUnitIds(int count)
            => _startedUnitIds.Take(count).ToArray();
    }

    private sealed class RecordingJobUnit(List<string> messages) : IJobUnit
    {
        public JobUnitOutcome? Outcome { get; private set; }

        public void Report(double progress, string? message = null)
        {
            if (!string.IsNullOrWhiteSpace(message))
                messages.Add(message);
        }

        public void Complete(JobUnitOutcome outcome, string? message = null)
        {
            Outcome ??= outcome;
            if (!string.IsNullOrWhiteSpace(message))
                messages.Add(message);
        }

        public void Dispose()
        {
        }
    }

    private sealed class PagingVideoRepository(IReadOnlyList<Video> videos) : IVideoRepository
    {
        public FindFilter? LastFindFilter { get; private set; }

        public Task<(IReadOnlyList<Video> Items, int TotalCount)> FindAsync(VideoFilter? filter, FindFilter? findFilter, CancellationToken ct = default)
        {
            LastFindFilter = findFilter;
            var ids = filter?.Ids ?? [];
            var page = findFilter?.Page ?? 1;
            var perPage = findFilter?.PerPage ?? 25;
            var items = videos
                .Where(video => ids.Count == 0 || ids.Contains(video.Id))
                .Skip((page - 1) * perPage)
                .Take(perPage)
                .ToList();
            return Task.FromResult<(IReadOnlyList<Video> Items, int TotalCount)>((items, ids.Count));
        }

        public Task<Video?> GetByIdWithRelationsAsync(int id, CancellationToken ct = default) => throw new NotSupportedException();

        public Task<IReadOnlyList<VideoPerformer>> GetVideoPerformersAsync(IReadOnlyList<int> videoIds, CancellationToken ct = default) => throw new NotSupportedException();

        public Task<Video?> GetByIdAsync(int id, CancellationToken ct = default) => throw new NotSupportedException();

        public Task<IReadOnlyList<Video>> GetAllAsync(CancellationToken ct = default) => throw new NotSupportedException();

        public Task<Video> AddAsync(Video entity, CancellationToken ct = default) => throw new NotSupportedException();

        public Task UpdateAsync(Video entity, CancellationToken ct = default) => throw new NotSupportedException();

        public Task DeleteAsync(int id, CancellationToken ct = default) => throw new NotSupportedException();

        public Task<int> CountAsync(CancellationToken ct = default) => throw new NotSupportedException();
    }

    private sealed class ThrowingImageRepository : IImageRepository
    {
        public Task<(IReadOnlyList<Image> Items, int TotalCount)> FindAsync(ImageFilter? filter, FindFilter? findFilter, CancellationToken ct = default) => throw new NotSupportedException();

        public Task<Image?> GetByIdWithRelationsAsync(int id, CancellationToken ct = default) => throw new NotSupportedException();

        public Task<IReadOnlyList<ImagePerformer>> GetImagePerformersAsync(IReadOnlyList<int> imageIds, CancellationToken ct = default) => throw new NotSupportedException();

        public Task<IReadOnlyList<int>> GetTagIdsAsync(int imageId, CancellationToken ct = default) => throw new NotSupportedException();

        public void AddTagLink(int imageId, int tagId) => throw new NotSupportedException();

        public Task<Image?> GetByIdAsync(int id, CancellationToken ct = default) => throw new NotSupportedException();

        public Task<IReadOnlyList<Image>> GetAllAsync(CancellationToken ct = default) => throw new NotSupportedException();

        public Task<Image> AddAsync(Image entity, CancellationToken ct = default) => throw new NotSupportedException();

        public Task UpdateAsync(Image entity, CancellationToken ct = default) => throw new NotSupportedException();

        public Task DeleteAsync(int id, CancellationToken ct = default) => throw new NotSupportedException();

        public Task<int> CountAsync(CancellationToken ct = default) => throw new NotSupportedException();
    }
}

using System.Text.Json;

using AI.Core;
using AI.Extensions.Abstractions;

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
}
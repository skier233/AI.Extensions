using System.Threading.Channels;

using AI.Extensions.Abstractions;

using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace AI.Faces;

/// <summary>
/// A single face-cover crop to generate: decode the source image, crop the face box, encode a JPEG thumbnail,
/// store it as the face's cover blob. Carries only what the work needs so it can be processed later, off the
/// thread that ran the AI-server request.
/// </summary>
internal sealed record FaceCoverWorkItem(
    string HostEntityType,
    int FaceId,
    AiPreparedFaceIdentity CoverFace,
    AiPreparedDetection? CoverDetection,
    double? IncomingCoverQuality);

internal interface IAiFaceCoverQueue
{
    /// <summary>Queues cover crops for background generation. Non-blocking.</summary>
    void Enqueue(IServiceProvider services, IReadOnlyList<FaceCoverWorkItem> items);
}

/// <summary>
/// Generates face cover thumbnails on a background pool sized to the host's <c>MaxParallelTasks</c> setting,
/// independent of the AI run's in-flight-request slots. Cover generation is CPU-bound image decode/crop/encode;
/// running it inline on an AI-request worker held that worker (and therefore an AI-server slot) for the whole
/// crop, starving the GPU. The dispatch now persists faces and <see cref="Enqueue"/>s the crops here, freeing
/// the worker to issue the next AI-server request immediately; covers fill in asynchronously.
/// </summary>
internal sealed class AiFaceCoverQueue(IServiceScopeFactory scopeFactory, ILogger<AiFaceCoverQueue> logger) : IAiFaceCoverQueue
{
    private readonly Channel<FaceCoverWorkItem> _channel = Channel.CreateUnbounded<FaceCoverWorkItem>(
        new UnboundedChannelOptions { SingleReader = false, SingleWriter = false });

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;
    private readonly ILogger<AiFaceCoverQueue> _logger = logger;
    private readonly Lock _startLock = new();
    private SemaphoreSlim? _gate;

    /// <summary>
    /// Queues a batch of cover crops for background generation. Non-blocking. The first call resolves the
    /// concurrency limit from <see cref="CoveConfiguration.MaxParallelTasks"/> via <paramref name="services"/>
    /// and starts the pump.
    /// </summary>
    public void Enqueue(IServiceProvider services, IReadOnlyList<FaceCoverWorkItem> items)
    {
        if (items.Count == 0)
        {
            return;
        }

        EnsureStarted(services);
        foreach (var item in items)
        {
            _channel.Writer.TryWrite(item);
        }
    }

    private void EnsureStarted(IServiceProvider services)
    {
        if (_gate is not null)
        {
            return;
        }

        lock (_startLock)
        {
            if (_gate is not null)
            {
                return;
            }

            var configured = services.GetService<CoveConfiguration>()?.MaxParallelTasks ?? 0;
            var maxParallel = configured <= 0 ? Environment.ProcessorCount : Math.Max(1, configured);
            _gate = new SemaphoreSlim(maxParallel);
            _ = Task.Run(PumpAsync);
            _logger.LogInformation("AI.Faces cover-generation pool started ({MaxParallel} parallel).", maxParallel);
        }
    }

    private async Task PumpAsync()
    {
        var gate = _gate!;
        await foreach (var item in _channel.Reader.ReadAllAsync())
        {
            await gate.WaitAsync();
            // Fire-and-forget so the pump immediately pulls the next item up to the parallelism limit.
            _ = ProcessAsync(item, gate);
        }
    }

    private async Task ProcessAsync(FaceCoverWorkItem item, SemaphoreSlim gate)
    {
        try
        {
            using var scope = _scopeFactory.CreateScope();
            // Host-lifetime token, not the originating job's: covers must finish even after the job completes.
            await AiFacesPersistenceService.GenerateAndStoreCoverAsync(scope.ServiceProvider, item, CancellationToken.None);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "AI.Faces background cover generation failed for face {FaceId}.", item.FaceId);
        }
        finally
        {
            gate.Release();
        }
    }
}

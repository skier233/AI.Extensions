using System.Diagnostics;
using System.Globalization;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Review;

// Extracts single frames with ffmpeg. Thumbnails for the review grid are cached on disk under the
// extension's data folder; full-resolution frames are written straight to the export folder.
internal sealed class AiReviewFrameService
{
    private static readonly TimeSpan FrameExtractionTimeout = TimeSpan.FromSeconds(40);

    private readonly IServiceScopeFactory _scopeFactory;
    private readonly CoveConfiguration? _configuration;
    private readonly SemaphoreSlim _gate;
    private readonly string _cacheRoot;

    public AiReviewFrameService(IServiceScopeFactory scopeFactory, CoveConfiguration? configuration, string dataRoot)
    {
        _scopeFactory = scopeFactory;
        _configuration = configuration;
        _cacheRoot = Path.Combine(dataRoot, "cache");
        var parallel = Math.Clamp(configuration?.MaxParallelTasks ?? 4, 1, 8);
        _gate = new SemaphoreSlim(parallel, parallel);
    }

    public string CacheRoot => _cacheRoot;

    private readonly HashSet<string> _warming = new(StringComparer.Ordinal);
    private readonly object _warmingLock = new();

    // Generates thumbnails for a page of segments in the background so that by the time the grid
    // scrolls to them they are already on disk. Duplicate requests are collapsed.
    public void Warmup(IEnumerable<(int VideoId, double Seconds)> frames, int width)
    {
        foreach (var (videoId, seconds) in frames)
        {
            var target = ThumbnailPath(videoId, seconds, width);
            if (File.Exists(target))
            {
                continue;
            }

            lock (_warmingLock)
            {
                if (!_warming.Add(target))
                {
                    continue;
                }
            }

            _ = Task.Run(async () =>
            {
                try
                {
                    await GetThumbnailAsync(videoId, seconds, width, CancellationToken.None);
                }
                catch
                {
                    // best effort; the grid's own request will retry and surface the failure
                }
                finally
                {
                    lock (_warmingLock)
                    {
                        _warming.Remove(target);
                    }
                }
            });
        }
    }

    public async Task<string?> ResolveVideoPathAsync(int videoId, CancellationToken ct)
    {
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();
        var video = await db.Set<Video>()
            .AsNoTracking()
            .Include(v => v.Files)
            .FirstOrDefaultAsync(v => v.Id == videoId, ct);
        return video is null ? null : SelectFilePath(video);
    }

    public static string? SelectFilePath(Video video)
    {
        var file = video.Files
            .Where(static f => !string.IsNullOrWhiteSpace(f.Path))
            .OrderByDescending(f => f.Id == video.PrimaryFileId)
            .ThenByDescending(static f => f.Duration)
            .ThenBy(static f => f.Path, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
        return file?.Path;
    }

    public string ThumbnailPath(int videoId, double seconds, int width)
    {
        var ms = (long)Math.Round(Math.Max(0, seconds) * 1000);
        return Path.Combine(_cacheRoot, videoId.ToString(CultureInfo.InvariantCulture), $"{ms:D9}_{width}.jpg");
    }

    // Returns the cached thumbnail path, extracting it first when needed. Null when the video has no
    // file on disk or ffmpeg failed.
    public async Task<string?> GetThumbnailAsync(int videoId, double seconds, int width, CancellationToken ct)
    {
        var target = ThumbnailPath(videoId, seconds, width);
        if (File.Exists(target))
        {
            return target;
        }

        var videoPath = await ResolveVideoPathAsync(videoId, ct);
        if (videoPath is null || !File.Exists(videoPath))
        {
            return null;
        }

        var ok = await ExtractFrameAsync(videoPath, seconds, target, width, quality: 4, ct);
        return ok ? target : null;
    }

    // Writes one JPEG frame to outputPath (atomically via a temp file). width == null keeps the source resolution.
    public async Task<bool> ExtractFrameAsync(string videoPath, double seconds, string outputPath, int? width, int quality, CancellationToken ct)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        var tempPath = outputPath + "." + Guid.NewGuid().ToString("N") + ".tmp.jpg";

        await _gate.WaitAsync(ct);
        try
        {
            if (File.Exists(outputPath))
            {
                return true;
            }

            var ok = await RunFfmpegAsync(videoPath, seconds, tempPath, width, quality, ct);
            if (!ok)
            {
                TryDelete(tempPath);
                return false;
            }

            try
            {
                File.Move(tempPath, outputPath, overwrite: true);
            }
            catch (IOException)
            {
                TryDelete(tempPath);
                return File.Exists(outputPath);
            }

            return true;
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task<bool> RunFfmpegAsync(string videoPath, double seconds, string outputPath, int? width, int quality, CancellationToken ct)
    {
        var scaleFilter = width is { } w && w > 0 ? $" -vf scale={w}:-2" : string.Empty;
        var seek = Math.Max(0, seconds).ToString("0.###", CultureInfo.InvariantCulture);
        var psi = new ProcessStartInfo
        {
            FileName = ResolveFfmpegPath(),
            Arguments = $"-hide_banner -loglevel error -nostdin -y -ss {seek} -i \"{videoPath}\" -an -sn -dn -frames:v 1{scaleFilter} -q:v {quality} -f image2 \"{outputPath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        Process? process;
        try
        {
            process = Process.Start(psi);
        }
        catch
        {
            return false;
        }

        if (process is null)
        {
            return false;
        }

        using (process)
        {
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeoutCts.CancelAfter(FrameExtractionTimeout);
            var stdoutTask = process.StandardOutput.ReadToEndAsync(timeoutCts.Token);
            var stderrTask = process.StandardError.ReadToEndAsync(timeoutCts.Token);
            try
            {
                await process.WaitForExitAsync(timeoutCts.Token);
                await stdoutTask;
                await stderrTask;
            }
            catch (OperationCanceledException)
            {
                try { process.Kill(entireProcessTree: true); } catch { }
                return false;
            }

            if (process.ExitCode != 0)
            {
                return false;
            }
        }

        return File.Exists(outputPath) && new FileInfo(outputPath).Length > 0;
    }

    private string ResolveFfmpegPath()
    {
        if (!string.IsNullOrWhiteSpace(_configuration?.FfmpegPath) && File.Exists(_configuration.FfmpegPath))
        {
            return _configuration.FfmpegPath;
        }

        return "ffmpeg";
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // best effort
        }
    }
}

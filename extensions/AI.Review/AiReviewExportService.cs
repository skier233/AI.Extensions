using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Review;

// Background exporter. For every reviewed segment it writes the planned frames to
//   <export root>/<tag folder>/<fp|tp>/v<videoId>_s<segmentId>_<ms>.jpg
// and keeps <export root>/manifest.csv in sync with the marks table. The manifest is the source of
// truth for training (it carries the exact tag name and verdict), so folder names only need to be
// human-readable.
internal sealed class AiReviewExportService
{
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly AiReviewSettingsProvider _settingsProvider;
    private readonly AiReviewFrameService _frameService;
    private readonly string _defaultExportRoot;
    private readonly Channel<int> _channel = Channel.CreateUnbounded<int>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
    private readonly SemaphoreSlim _manifestGate = new(1, 1);
    private readonly object _startLock = new();
    private CancellationTokenSource? _cts;
    private Task? _pump;
    private int _manifestDirty;

    public AiReviewExportService(IServiceScopeFactory scopeFactory, AiReviewSettingsProvider settingsProvider, AiReviewFrameService frameService, string dataRoot)
    {
        _scopeFactory = scopeFactory;
        _settingsProvider = settingsProvider;
        _frameService = frameService;
        _defaultExportRoot = Path.Combine(dataRoot, "export");
    }

    public string ResolveExportRoot(AiReviewSettings settings)
        => string.IsNullOrWhiteSpace(settings.ExportRootPath) ? _defaultExportRoot : settings.ExportRootPath;

    public string ManifestPath(AiReviewSettings settings) => Path.Combine(ResolveExportRoot(settings), "manifest.csv");

    public void Start()
    {
        lock (_startLock)
        {
            if (_pump is not null)
            {
                return;
            }

            _cts = new CancellationTokenSource();
            _pump = Task.Run(() => PumpAsync(_cts.Token));
        }
    }

    public void Stop()
    {
        lock (_startLock)
        {
            _cts?.Cancel();
            _channel.Writer.TryComplete();
        }
    }

    public void Enqueue(int markId)
    {
        Start();
        _channel.Writer.TryWrite(markId);
    }

    public void RequestManifestRewrite()
    {
        Interlocked.Exchange(ref _manifestDirty, 1);
        Enqueue(0);
    }

    // Re-queue everything that is not done yet (called at startup and from the "retry" button).
    public async Task<int> EnqueuePendingAsync(bool includeFailed, CancellationToken ct)
    {
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();
        string[] statuses = includeFailed
            ? [AiReviewExportStatus.Pending, AiReviewExportStatus.Failed]
            : [AiReviewExportStatus.Pending];
        var ids = await db.Set<ExtAiReviewMarkEntity>().AsNoTracking()
            .Where(m => statuses.Contains(m.ExportStatus))
            .Select(m => m.Id)
            .ToListAsync(ct);
        foreach (var id in ids)
        {
            Enqueue(id);
        }

        return ids.Count;
    }

    public async Task<AiReviewExportStatusDto> GetStatusAsync(CancellationToken ct)
    {
        var settings = await _settingsProvider.GetAsync(ct);
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();
        var counts = await db.Set<ExtAiReviewMarkEntity>().AsNoTracking()
            .GroupBy(m => m.ExportStatus)
            .Select(g => new { Status = g.Key, Count = g.Count() })
            .ToListAsync(ct);
        int Count(string status) => counts.FirstOrDefault(c => c.Status == status)?.Count ?? 0;
        return new AiReviewExportStatusDto(
            Count(AiReviewExportStatus.Pending),
            Count(AiReviewExportStatus.Done),
            Count(AiReviewExportStatus.Failed),
            Count(AiReviewExportStatus.Skipped),
            ResolveExportRoot(settings),
            ManifestPath(settings));
    }

    private async Task PumpAsync(CancellationToken ct)
    {
        try
        {
            await foreach (var markId in _channel.Reader.ReadAllAsync(ct))
            {
                try
                {
                    if (markId > 0)
                    {
                        await ExportMarkAsync(markId, ct);
                        Interlocked.Exchange(ref _manifestDirty, 1);
                    }

                    // Rewrite the manifest once the queue drains (or after each item when idle).
                    if (_channel.Reader.Count == 0 && Interlocked.Exchange(ref _manifestDirty, 0) == 1)
                    {
                        await RewriteManifestAsync(ct);
                    }
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    throw;
                }
                catch
                {
                    // Per-item failures are recorded on the mark; never let one stop the pump.
                }
            }
        }
        catch (OperationCanceledException)
        {
            // shutting down
        }
    }

    public async Task ExportMarkAsync(int markId, CancellationToken ct)
    {
        var settings = await _settingsProvider.GetAsync(ct);
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();
        var mark = await db.Set<ExtAiReviewMarkEntity>().FirstOrDefaultAsync(m => m.Id == markId, ct);
        if (mark is null)
        {
            return;
        }

        if (mark.Verdict == AiReviewVerdicts.TruePositive && !settings.ExportTruePositives)
        {
            DeleteExportedFiles(AiReviewQueryService.ReadStringList(mark.ExportedFilesJson));
            mark.ExportedFilesJson = "[]";
            mark.ExportStatus = AiReviewExportStatus.Skipped;
            mark.ExportError = null;
            mark.UpdatedAt = DateTime.UtcNow;
            await db.SaveChangesAsync(ct);
            return;
        }

        var frameTimes = AiReviewQueryService.ReadFrameTimes(mark.FrameTimesJson);
        if (frameTimes.Count == 0)
        {
            frameTimes = AiReviewFramePlanner.Plan(mark.StartSec, mark.EndSec, 0, settings.MaxFramesPerSegment);
            mark.FrameTimesJson = JsonSerializer.Serialize(frameTimes);
        }

        var videoPath = await _frameService.ResolveVideoPathAsync(mark.VideoId, ct);
        if (videoPath is null || !File.Exists(videoPath))
        {
            mark.ExportStatus = AiReviewExportStatus.Failed;
            mark.ExportError = "Video file not found on disk.";
            mark.UpdatedAt = DateTime.UtcNow;
            await db.SaveChangesAsync(ct);
            return;
        }

        var targetDir = Path.Combine(ResolveExportRoot(settings), TagFolderName(mark.TagName), mark.Verdict);
        Directory.CreateDirectory(targetDir);

        // Frames exported under a previous verdict are stale (different folder): remove them first.
        var previous = AiReviewQueryService.ReadStringList(mark.ExportedFilesJson)
            .Where(path => !path.StartsWith(targetDir, StringComparison.OrdinalIgnoreCase))
            .ToList();
        DeleteExportedFiles(previous);

        var written = new List<string>();
        var failures = new List<string>();
        var segmentId = mark.SegmentId ?? 0;
        foreach (var time in frameTimes)
        {
            var ms = (long)Math.Round(time * 1000);
            var fileName = $"v{mark.VideoId}_s{(segmentId > 0 ? segmentId.ToString(CultureInfo.InvariantCulture) : "m" + mark.Id)}_{ms:D9}.jpg";
            var outputPath = Path.Combine(targetDir, fileName);
            var ok = await _frameService.ExtractFrameAsync(videoPath, time, outputPath, settings.ExportFullResolution ? null : settings.ThumbnailWidth, quality: 2, ct);
            if (ok)
            {
                written.Add(outputPath);
            }
            else
            {
                failures.Add(time.ToString("0.###", CultureInfo.InvariantCulture));
            }
        }

        mark.ExportedFilesJson = JsonSerializer.Serialize(written);
        mark.ExportedAt = DateTime.UtcNow;
        mark.UpdatedAt = DateTime.UtcNow;
        if (written.Count == 0)
        {
            mark.ExportStatus = AiReviewExportStatus.Failed;
            mark.ExportError = "ffmpeg could not extract any frame (times: " + string.Join(", ", failures) + ").";
        }
        else
        {
            mark.ExportStatus = AiReviewExportStatus.Done;
            mark.ExportError = failures.Count > 0 ? "Some frames failed: " + string.Join(", ", failures) : null;
        }

        await db.SaveChangesAsync(ct);
    }

    public async Task RewriteManifestAsync(CancellationToken ct)
    {
        var settings = await _settingsProvider.GetAsync(ct);
        var manifestPath = ManifestPath(settings);
        await _manifestGate.WaitAsync(ct);
        try
        {
            using var scope = _scopeFactory.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<DbContext>();
            var marks = await db.Set<ExtAiReviewMarkEntity>().AsNoTracking()
                .Where(m => m.ExportStatus == AiReviewExportStatus.Done)
                .OrderBy(m => m.TagName).ThenBy(m => m.VideoId).ThenBy(m => m.StartSec)
                .ToListAsync(ct);
            var videoIds = marks.Select(m => m.VideoId).Distinct().ToList();
            var titles = await db.Set<Cove.Core.Entities.Video>().AsNoTracking()
                .Where(v => videoIds.Contains(v.Id))
                .Select(v => new { v.Id, v.Title })
                .ToDictionaryAsync(v => v.Id, v => v.Title ?? string.Empty, ct);

            var sb = new StringBuilder();
            sb.AppendLine("file,tag,verdict,label,video_id,video_title,segment_id,model_key,frame_sec,segment_start_sec,segment_end_sec,confidence,reviewer,reviewed_at");
            foreach (var mark in marks)
            {
                var files = AiReviewQueryService.ReadStringList(mark.ExportedFilesJson);
                foreach (var file in files)
                {
                    var frameSec = ParseFrameSeconds(file);
                    sb.Append(Csv(file)).Append(',')
                      .Append(Csv(mark.TagName)).Append(',')
                      .Append(mark.Verdict).Append(',')
                      .Append(mark.Verdict == AiReviewVerdicts.TruePositive ? "1" : "0").Append(',')
                      .Append(mark.VideoId.ToString(CultureInfo.InvariantCulture)).Append(',')
                      .Append(Csv(titles.GetValueOrDefault(mark.VideoId, string.Empty))).Append(',')
                      .Append(mark.SegmentId?.ToString(CultureInfo.InvariantCulture) ?? string.Empty).Append(',')
                      .Append(Csv(mark.ModelKey ?? string.Empty)).Append(',')
                      .Append(frameSec?.ToString("0.###", CultureInfo.InvariantCulture) ?? string.Empty).Append(',')
                      .Append(mark.StartSec.ToString("0.###", CultureInfo.InvariantCulture)).Append(',')
                      .Append(mark.EndSec.ToString("0.###", CultureInfo.InvariantCulture)).Append(',')
                      .Append(mark.Confidence?.ToString("0.####", CultureInfo.InvariantCulture) ?? string.Empty).Append(',')
                      .Append(Csv(mark.ReviewerName ?? string.Empty)).Append(',')
                      .Append(mark.UpdatedAt.ToString("o", CultureInfo.InvariantCulture))
                      .AppendLine();
                }
            }

            Directory.CreateDirectory(Path.GetDirectoryName(manifestPath)!);
            var temp = manifestPath + ".tmp";
            await File.WriteAllTextAsync(temp, sb.ToString(), new UTF8Encoding(false), ct);
            File.Move(temp, manifestPath, overwrite: true);
        }
        finally
        {
            _manifestGate.Release();
        }
    }

    internal static double? ParseFrameSeconds(string file)
    {
        var name = Path.GetFileNameWithoutExtension(file);
        var underscore = name.LastIndexOf('_');
        if (underscore < 0 || !long.TryParse(name[(underscore + 1)..], NumberStyles.None, CultureInfo.InvariantCulture, out var ms))
        {
            return null;
        }

        return ms / 1000.0;
    }

    // Lowercase tag name with characters that cannot appear in a folder name replaced. The manifest
    // keeps the exact tag name, so this only needs to be readable and unique enough.
    internal static string TagFolderName(string tagName)
    {
        var trimmed = (tagName ?? string.Empty).Trim().ToLowerInvariant();
        if (trimmed.Length == 0)
        {
            return "_unknown";
        }

        var invalid = Path.GetInvalidFileNameChars();
        var chars = trimmed.Select(c => invalid.Contains(c) || c == '/' || c == '\\' ? '_' : c).ToArray();
        var result = new string(chars).Trim('.', ' ');
        return result.Length == 0 ? "_unknown" : result;
    }

    internal static string Csv(string value)
    {
        if (value.IndexOfAny([',', '"', '\n', '\r']) < 0)
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    internal static void DeleteExportedFiles(IEnumerable<string> paths)
    {
        foreach (var path in paths)
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
}

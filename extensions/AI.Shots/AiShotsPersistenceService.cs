using Cove.Core.DTOs;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

namespace AI.Shots;

/// <summary>One file of a video, with the duration Cove probed for it.</summary>
internal sealed record VideoFileInfo(int FileId, string Path, double Duration);

/// <summary>The files of a video, for resolving which one an analysis ran on.</summary>
internal interface IAiShotsFileLookup
{
    Task<IReadOnlyList<VideoFileInfo>> GetVideoFilesAsync(int videoId, CancellationToken ct = default);
}

internal sealed class VideoRepositoryFileLookup(IVideoRepository videoRepository) : IAiShotsFileLookup
{
    private readonly IVideoRepository _videoRepository = videoRepository;

    public async Task<IReadOnlyList<VideoFileInfo>> GetVideoFilesAsync(int videoId, CancellationToken ct = default)
    {
        var (videos, _) = await _videoRepository.FindAsync(
            new VideoFilter { Ids = [videoId] },
            new FindFilter { PerPage = 1 },
            ct);
        return videos
            .Where(video => video.Id == videoId)
            .SelectMany(static video => video.Files)
            .Where(static file => !string.IsNullOrWhiteSpace(file.Path))
            .Select(static file => new VideoFileInfo(file.Id, file.Path, file.Duration))
            .ToArray();
    }
}

/// <summary>Reads and writes shot boundaries through Cove's <see cref="IVideoShotService"/>, one scope per call.</summary>
internal sealed class AiShotsPersistenceService(IServiceScopeFactory scopeFactory)
{
    public const string SourceKey = "ext:ai.shots";

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;

    /// <summary>The file's existing set summary, or null when it has none.</summary>
    public async Task<VideoShotSetDto?> GetExistingSetAsync(int fileId, CancellationToken ct = default)
    {
        await using var scope = _scopeFactory.CreateAsyncScope();
        var shots = scope.ServiceProvider.GetRequiredService<IVideoShotService>();
        var summaries = await shots.GetSummariesForFilesAsync([fileId], ct);
        return summaries.TryGetValue(fileId, out var set) ? set : null;
    }

    public async Task<VideoShotWriteResult> WriteAsync(VideoShotSetWrite write, CancellationToken ct = default)
    {
        await using var scope = _scopeFactory.CreateAsyncScope();
        var shots = scope.ServiceProvider.GetRequiredService<IVideoShotService>();
        return await shots.WriteSetAsync(write, ct);
    }

    /// <summary>
    /// The file of <paramref name="videoId"/> the analysis ran on: the file AI Core named, when it belongs to the video,
    /// otherwise the video's one file whose path is <paramref name="subject"/>, the path AI Core analysed. Null when
    /// neither identifies exactly one file of the video.
    /// </summary>
    public async Task<VideoFileInfo?> ResolveFileAsync(int videoId, int? namedFileId, string? subject, CancellationToken ct = default)
    {
        await using var scope = _scopeFactory.CreateAsyncScope();
        var lookup = scope.ServiceProvider.GetRequiredService<IAiShotsFileLookup>();
        var files = await lookup.GetVideoFilesAsync(videoId, ct);
        if (namedFileId is int fileId && files.FirstOrDefault(file => file.FileId == fileId) is { } named)
        {
            return named;
        }

        var normalized = NormalizePath(subject);
        if (normalized.Length == 0)
        {
            return null;
        }

        var exact = files.Where(file => string.Equals(NormalizePath(file.Path), normalized, StringComparison.Ordinal)).ToArray();
        if (exact.Length == 1)
        {
            return exact[0];
        }

        var ignoringCase = files.Where(file => string.Equals(NormalizePath(file.Path), normalized, StringComparison.OrdinalIgnoreCase)).ToArray();
        return ignoringCase.Length == 1 ? ignoringCase[0] : null;
    }

    // Forward slashes, no trailing slash, and a slash after a drive letter, as AI Core normalises the paths it analyses.
    private static string NormalizePath(string? path)
    {
        var normalized = (path ?? string.Empty).Trim().Replace('\\', '/').TrimEnd('/');
        if (normalized.Length >= 2 && char.IsLetter(normalized[0]) && normalized[1] == ':' && (normalized.Length == 2 || normalized[2] != '/'))
        {
            normalized = normalized[..2] + "/" + normalized[2..];
        }

        return normalized;
    }
}

using System.Diagnostics;
using System.Globalization;

using AI.Extensions.Abstractions;

using Cove.Core.Entities.Galleries.Zip;
using Cove.Core.Interfaces;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AI.Faces;

// Where a face cover is cropped from: the frame at ObservedAtSeconds of the video at Path, or, without a time, the
// still at Path (an image file or a "<archive>.zip/<entry>" gallery image). BoundingBox was detected on that frame
// or still.
internal sealed record FaceCoverSource(string Path, AiBoundingBox BoundingBox, double? ObservedAtSeconds);

// A built cover, or why none could be built.
internal readonly record struct FaceCoverResult(MemoryStream? Cover, string? FailureReason)
{
    public static FaceCoverResult Failed(string reason) => new(null, reason);
}

internal static class AiFaceCoverGenerator
{
    private const int FaceCoverSize = 768;
    private const int FaceCoverQuality = 90;
    private const double CoverScale = 1.8;
    private const double VerticalCenterBias = 0.1;
    private const int VideoFrameExtractWidth = 1280;
    private const double BlankMeanLumaThreshold = 8.0;
    private const double BlankLumaVarianceThreshold = 12.0;
    private static readonly TimeSpan FrameExtractionTimeout = TimeSpan.FromSeconds(30);
    private static readonly string[] ArchiveExtensions = [".zip", ".cbz"];
    private static readonly HashSet<string> StillImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".jfif", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".tga", ".qoi", ".avif", ".heic", ".heif", ".jxl",
    };
    private static readonly IZipFileReader DefaultZipFileReader = new ZipFileReader();

    // Decodes the source the caller resolved: the frame at ObservedAtSeconds when that is set (a video), otherwise
    // the still at Path, which may be an "<archive>.zip/<entry>" gallery image.
    public static async Task<FaceCoverResult> CreateAsync(
        FaceCoverSource source,
        CoveConfiguration? configuration,
        IZipFileReader? zipFileReader,
        CancellationToken ct = default)
    {
        var assetPath = source.Path.Trim();
        if (File.Exists(assetPath))
        {
            return source.ObservedAtSeconds is { } seconds
                ? await CreateFromVideoAsync(assetPath, seconds, source.BoundingBox, configuration, ct)
                : await CreateFromImageAsync(assetPath, source.BoundingBox, ct);
        }

        if (TryParseArchivePath(assetPath, out var archivePath, out var entryPath))
        {
            return await CreateFromArchiveEntryAsync(zipFileReader ?? DefaultZipFileReader, archivePath, entryPath, source.BoundingBox, ct);
        }

        return FaceCoverResult.Failed("the source file does not exist");
    }

    // True when CreateAsync can read the cover source: a plain file, or an entry inside an archive that exists.
    internal static bool SourceExists(string path)
        => File.Exists(path) || TryParseArchivePath(path, out _, out _);

    // True when the path names a still image (a gallery archive entry or an image file) rather than a video, judged
    // from the path alone so a missing file is still classified.
    internal static bool IsStillImagePath(string path)
    {
        var normalizedPath = path.Replace('\\', '/');
        foreach (var extension in ArchiveExtensions)
        {
            var markerIndex = normalizedPath.IndexOf(extension + "/", StringComparison.OrdinalIgnoreCase);
            if (markerIndex >= 0 && normalizedPath[(markerIndex + extension.Length + 1)..].Trim('/').Length > 0) return true;
        }

        return StillImageExtensions.Contains(Path.GetExtension(normalizedPath));
    }

    private static async Task<FaceCoverResult> CreateFromImageAsync(string imagePath, AiBoundingBox boundingBox, CancellationToken ct)
    {
        try
        {
            await using var stream = new FileStream(imagePath, FileMode.Open, FileAccess.Read, FileShare.Read, 81920, useAsync: true);
            return await CreateFromStreamAsync(stream, boundingBox, ct);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return FaceCoverResult.Failed($"the image could not be read: {ex.Message}");
        }
    }

    private static async Task<FaceCoverResult> CreateFromArchiveEntryAsync(IZipFileReader zipFileReader, string archivePath, string entryPath, AiBoundingBox boundingBox, CancellationToken ct)
    {
        try
        {
            await using var stream = await zipFileReader.ExtractEntryAsync(archivePath, entryPath, ct);
            return await CreateFromStreamAsync(stream, boundingBox, ct);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            return FaceCoverResult.Failed($"the archive entry could not be read: {ex.Message}");
        }
    }

    // Mirrors Cove's ThumbnailService.TryParseArchivePath, which resolves the same "<archive>.zip/<entry>" image paths.
    private static bool TryParseArchivePath(string path, out string archivePath, out string entryPath)
    {
        archivePath = string.Empty;
        entryPath = string.Empty;

        var normalizedPath = path.Replace('\\', '/');
        foreach (var extension in ArchiveExtensions)
        {
            var markerIndex = normalizedPath.IndexOf(extension + "/", StringComparison.OrdinalIgnoreCase);
            if (markerIndex < 0) continue;

            var archiveEnd = markerIndex + extension.Length;
            var candidateArchivePath = path[..archiveEnd];
            var candidateEntryPath = normalizedPath[(archiveEnd + 1)..].Trim('/');
            if (candidateEntryPath.Length == 0 || !File.Exists(candidateArchivePath)) continue;

            archivePath = candidateArchivePath;
            entryPath = candidateEntryPath;
            return true;
        }

        return false;
    }

    private static async Task<FaceCoverResult> CreateFromVideoAsync(string videoPath, double seconds, AiBoundingBox boundingBox, CoveConfiguration? configuration, CancellationToken ct)
    {
        var ffmpegPath = ResolveFfmpegPath(configuration);
        var process = StartFrameExtraction(ffmpegPath, videoPath, seconds, downscale: IsNormalized(RepairBoundingBox(boundingBox)));
        if (process is null)
        {
            return FaceCoverResult.Failed($"ffmpeg ({ffmpegPath}) could not be started");
        }

        try
        {
            using (process)
            {
                await using var frameBuffer = new MemoryStream();
                using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                timeoutCts.CancelAfter(FrameExtractionTimeout);
                var stdoutTask = process.StandardOutput.BaseStream.CopyToAsync(frameBuffer, timeoutCts.Token);
                var stderrTask = process.StandardError.ReadToEndAsync(timeoutCts.Token);
                string stderr;

                try
                {
                    await process.WaitForExitAsync(timeoutCts.Token);
                    await stdoutTask;
                    stderr = await stderrTask;
                }
                catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                {
                    try { process.Kill(entireProcessTree: true); } catch { }
                    return FaceCoverResult.Failed($"ffmpeg timed out extracting the frame at {FormatSeconds(seconds)} s");
                }

                if (process.ExitCode != 0)
                {
                    return FaceCoverResult.Failed($"ffmpeg exited with code {process.ExitCode} extracting the frame at {FormatSeconds(seconds)} s: {stderr.Trim()}");
                }

                if (frameBuffer.Length == 0)
                {
                    return FaceCoverResult.Failed($"ffmpeg returned no frame at {FormatSeconds(seconds)} s");
                }

                frameBuffer.Position = 0;
                return await CreateFromStreamAsync(frameBuffer, boundingBox, ct);
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return FaceCoverResult.Failed($"the video frame could not be extracted: {ex.Message}");
        }
    }

    private static string FormatSeconds(double seconds) => seconds.ToString("0.###", CultureInfo.InvariantCulture);

    private static Process? StartFrameExtraction(string ffmpegPath, string videoPath, double seconds, bool downscale)
    {
        try
        {
            var scaleFilter = downscale ? $" -vf scale={VideoFrameExtractWidth}:-2" : string.Empty;
            var psi = new ProcessStartInfo
            {
                FileName = ffmpegPath,
                Arguments = $"-hide_banner -loglevel error -nostdin -ss {seconds.ToString("0.###", CultureInfo.InvariantCulture)} -i \"{videoPath}\" -an -sn -dn -frames:v 1{scaleFilter} -q:v 3 -f image2pipe -vcodec mjpeg pipe:1",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };

            return Process.Start(psi);
        }
        catch
        {
            return null;
        }
    }

    private static string ResolveFfmpegPath(CoveConfiguration? configuration)
    {
        if (!string.IsNullOrWhiteSpace(configuration?.FfmpegPath) && File.Exists(configuration.FfmpegPath))
        {
            return configuration.FfmpegPath;
        }

        return "ffmpeg";
    }

    private static async Task<FaceCoverResult> CreateFromStreamAsync(Stream sourceStream, AiBoundingBox boundingBox, CancellationToken ct)
    {
        try
        {
            using var image = await Image.LoadAsync<Rgba32>(sourceStream, ct);
            image.Mutate(static context => context.AutoOrient());
            if (IsNearlyBlankOrBlack(image))
            {
                return FaceCoverResult.Failed("the source image is blank or black");
            }

            var cropRect = BuildCropRectangle(image.Width, image.Height, boundingBox);
            image.Mutate(context => context.Crop(cropRect));
            if (IsNearlyBlankOrBlack(image))
            {
                return FaceCoverResult.Failed("the face crop is blank or black");
            }

            if (image.Width != FaceCoverSize || image.Height != FaceCoverSize)
            {
                image.Mutate(context => context.Resize(new ResizeOptions
                {
                    Mode = ResizeMode.Crop,
                    Position = AnchorPositionMode.Center,
                    Size = new Size(FaceCoverSize, FaceCoverSize),
                }));
            }

            var output = new MemoryStream();
            await image.SaveAsJpegAsync(output, new JpegEncoder { Quality = FaceCoverQuality }, ct);
            output.Position = 0;
            return new FaceCoverResult(output, null);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return FaceCoverResult.Failed($"the image could not be decoded: {ex.Message}");
        }
    }

    private static Rectangle BuildCropRectangle(int imageWidth, int imageHeight, AiBoundingBox boundingBox)
    {
        var repairedBoundingBox = RepairBoundingBox(boundingBox);
        var normalized = IsNormalized(repairedBoundingBox);

        var x1 = normalized ? repairedBoundingBox.X1 * imageWidth : repairedBoundingBox.X1;
        var y1 = normalized ? repairedBoundingBox.Y1 * imageHeight : repairedBoundingBox.Y1;
        var x2 = normalized ? repairedBoundingBox.X2 * imageWidth : repairedBoundingBox.X2;
        var y2 = normalized ? repairedBoundingBox.Y2 * imageHeight : repairedBoundingBox.Y2;

        var left = Clamp((int)Math.Floor(Math.Min(x1, x2)), 0, imageWidth - 1);
        var top = Clamp((int)Math.Floor(Math.Min(y1, y2)), 0, imageHeight - 1);
        var right = Clamp((int)Math.Ceiling(Math.Max(x1, x2)), left + 1, imageWidth);
        var bottom = Clamp((int)Math.Ceiling(Math.Max(y1, y2)), top + 1, imageHeight);

        var faceWidth = Math.Max(1, right - left);
        var faceHeight = Math.Max(1, bottom - top);
        var side = (int)Math.Ceiling(Math.Max(faceWidth, faceHeight) * CoverScale);
        side = Math.Min(side, Math.Min(imageWidth, imageHeight));
        side = Math.Max(1, side);

        var centerX = left + (faceWidth / 2.0);
        var centerY = top + (faceHeight / 2.0) - (faceHeight * VerticalCenterBias);

        var cropLeft = (int)Math.Round(centerX - (side / 2.0));
        var cropTop = (int)Math.Round(centerY - (side / 2.0));

        cropLeft = Clamp(cropLeft, 0, Math.Max(0, imageWidth - side));
        cropTop = Clamp(cropTop, 0, Math.Max(0, imageHeight - side));

        return new Rectangle(cropLeft, cropTop, Math.Min(side, imageWidth), Math.Min(side, imageHeight));
    }

    private static AiBoundingBox RepairBoundingBox(AiBoundingBox boundingBox)
    {
        var repairedX2 = boundingBox.X2 > boundingBox.X1
            ? boundingBox.X2
            : boundingBox.X1 + Math.Max(boundingBox.X2, 0.0);
        var repairedY2 = boundingBox.Y2 > boundingBox.Y1
            ? boundingBox.Y2
            : boundingBox.Y1 + Math.Max(boundingBox.Y2, 0.0);

        return new AiBoundingBox(
            boundingBox.X1,
            boundingBox.Y1,
            repairedX2,
            repairedY2);
    }

    private static bool IsNormalized(AiBoundingBox boundingBox)
        => boundingBox.X1 >= 0.0
           && boundingBox.Y1 >= 0.0
           && boundingBox.X2 <= 1.000001
           && boundingBox.Y2 <= 1.000001;

    private static bool IsNearlyBlankOrBlack(Image<Rgba32> image)
    {
        var stepX = Math.Max(1, image.Width / 64);
        var stepY = Math.Max(1, image.Height / 64);
        var count = 0;
        var sum = 0.0;
        var sumSquares = 0.0;

        for (var y = 0; y < image.Height; y += stepY)
        {
            var row = image.DangerousGetPixelRowMemory(y).Span;
            for (var x = 0; x < image.Width; x += stepX)
            {
                var pixel = row[x];
                var luma = (0.2126 * pixel.R) + (0.7152 * pixel.G) + (0.0722 * pixel.B);
                sum += luma;
                sumSquares += luma * luma;
                count++;
            }
        }

        if (count == 0)
        {
            return true;
        }

        var mean = sum / count;
        var variance = Math.Max(0.0, (sumSquares / count) - (mean * mean));
        return mean <= BlankMeanLumaThreshold && variance <= BlankLumaVarianceThreshold;
    }

    private static int Clamp(int value, int min, int max)
        => value < min ? min : value > max ? max : value;
}
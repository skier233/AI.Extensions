using System.Diagnostics;
using System.Globalization;

using AI.Extensions.Abstractions;

using Cove.Core.Interfaces;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AI.Faces;

internal static class AiFaceCoverGenerator
{
    private const int FaceCoverSize = 768;
    private const int FaceCoverQuality = 90;
    private const double CoverScale = 1.8;
    private const double VerticalCenterBias = 0.1;
    private const int VideoFrameExtractWidth = 1280;
    private static readonly TimeSpan FrameExtractionTimeout = TimeSpan.FromSeconds(30);

    public static async Task<MemoryStream?> CreateAsync(
        string hostEntityType,
        AiPreparedFaceIdentity preparedFace,
        AiPreparedDetection? coverDetection,
        CoveConfiguration? configuration,
        CancellationToken ct = default)
    {
        if (preparedFace.CoverBoundingBox is null || string.IsNullOrWhiteSpace(preparedFace.CoverAssetId))
        {
            return null;
        }

        var coverBoundingBox = preparedFace.CoverBoundingBox.Value;
        var assetPath = preparedFace.CoverAssetId.Trim();
        if (!File.Exists(assetPath))
        {
            return null;
        }

        return hostEntityType switch
        {
            "scene" when coverDetection?.ObservedAtSeconds is { } seconds => await CreateFromVideoAsync(assetPath, seconds, coverBoundingBox, configuration, ct),
            "image" => await CreateFromImageAsync(assetPath, coverBoundingBox, ct),
            _ => null,
        };
    }

    private static async Task<MemoryStream?> CreateFromImageAsync(string imagePath, AiBoundingBox boundingBox, CancellationToken ct)
    {
        await using var stream = new FileStream(imagePath, FileMode.Open, FileAccess.Read, FileShare.Read, 81920, useAsync: true);
        return await CreateFromStreamAsync(stream, boundingBox, ct);
    }

    private static async Task<MemoryStream?> CreateFromVideoAsync(string videoPath, double seconds, AiBoundingBox boundingBox, CoveConfiguration? configuration, CancellationToken ct)
    {
        var ffmpegPath = ResolveFfmpegPath(configuration);
        var process = StartFrameExtraction(ffmpegPath, videoPath, seconds, downscale: IsNormalized(RepairBoundingBox(boundingBox)));
        if (process is null)
        {
            return null;
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

                try
                {
                    await process.WaitForExitAsync(timeoutCts.Token);
                    await stdoutTask;
                    await stderrTask;
                }
                catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                {
                    try { process.Kill(entireProcessTree: true); } catch { }
                    return null;
                }

                if (process.ExitCode != 0 || frameBuffer.Length == 0)
                {
                    return null;
                }

                frameBuffer.Position = 0;
                return await CreateFromStreamAsync(frameBuffer, boundingBox, ct);
            }
        }
        catch
        {
            return null;
        }
    }

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

    private static async Task<MemoryStream?> CreateFromStreamAsync(Stream sourceStream, AiBoundingBox boundingBox, CancellationToken ct)
    {
        try
        {
            using var image = await Image.LoadAsync<Rgba32>(sourceStream, ct);
            image.Mutate(static context => context.AutoOrient());

            var cropRect = BuildCropRectangle(image.Width, image.Height, boundingBox);
            image.Mutate(context => context.Crop(cropRect));
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
            return output;
        }
        catch
        {
            return null;
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

    private static int Clamp(int value, int min, int max)
        => value < min ? min : value > max ? max : value;
}
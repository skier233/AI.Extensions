using System.Diagnostics;
using System.Globalization;

using SixLabors.ImageSharp.PixelFormats;

using Xunit;

namespace AI.Extensions.Tests;

// Runs the test only where ffmpeg is on PATH. Cover crops from video frames shell out to ffmpeg, so these tests
// exercise the real seek; elsewhere they are reported as skipped instead of passing silently.
internal sealed class FfmpegFactAttribute : FactAttribute
{
    public FfmpegFactAttribute()
    {
        if (!TestVideos.FfmpegAvailable)
        {
            Skip = "ffmpeg is not on PATH.";
        }
    }
}

internal static class TestVideos
{
    private static readonly Lazy<bool> Available = new(static () => Run("-hide_banner -version") == 0);

    public static bool FfmpegAvailable => Available.Value;

    // Writes a small video that shows one solid color until switchAtSeconds and another after it, so a crop
    // taken from it shows which moment ffmpeg seeked to.
    public static void CreateTwoColorVideo(string path, Rgba32 first, Rgba32 second, double switchAtSeconds, double totalSeconds)
    {
        var firstDuration = switchAtSeconds.ToString("0.###", CultureInfo.InvariantCulture);
        var secondDuration = (totalSeconds - switchAtSeconds).ToString("0.###", CultureInfo.InvariantCulture);
        var arguments = "-hide_banner -loglevel error -y "
            + $"-f lavfi -i color=c={Hex(first)}:s=320x240:r=10:d={firstDuration} "
            + $"-f lavfi -i color=c={Hex(second)}:s=320x240:r=10:d={secondDuration} "
            + "-filter_complex \"[0:v][1:v]concat=n=2:v=1:a=0[v]\" -map \"[v]\" -pix_fmt yuv420p "
            + $"\"{path}\"";
        var exitCode = Run(arguments);
        if (exitCode != 0 || !File.Exists(path))
        {
            throw new InvalidOperationException($"ffmpeg could not create the test video (exit code {exitCode}).");
        }
    }

    private static string Hex(Rgba32 color) => $"0x{color.R:X2}{color.G:X2}{color.B:X2}";

    private static int Run(string arguments)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "ffmpeg",
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            });
            if (process is null)
            {
                return -1;
            }

            process.StandardOutput.ReadToEnd();
            process.StandardError.ReadToEnd();
            process.WaitForExit();
            return process.ExitCode;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return -1;
        }
    }
}

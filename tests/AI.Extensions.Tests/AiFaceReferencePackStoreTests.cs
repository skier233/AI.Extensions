using System.IO.Compression;
using System.Text;
using System.Text.Json;

using AI.Faces;

using Cove.Core.Interfaces;
using Cove.Plugins;

using CoreJobProgress = Cove.Core.Interfaces.IJobProgress;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiFaceReferencePackStoreTests
{
    [Fact]
    public async Task ImportStagedAsync_PersistsStatusAndCachesTheActivePack()
    {
        var tempRoot = Path.Combine(Path.GetTempPath(), $"ai-faces-reference-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);

        try
        {
            var archivePath = CreateArchive(
                tempRoot,
                "stashdb_server_20260323_202719.saie",
                new
                {
                    version = 1,
                    embedder = "nsfw_ai_server:image_pipeline_dynamic_v4",
                    embedding_dim = 4,
                    pack_id = "stashdb_server_20260323_202719",
                    source_endpoint = "https://stashdb.org/graphql",
                    performer_count = 2,
                    created_at = "2026-03-24T02:43:03Z",
                },
                [
                    new
                    {
                        stashdb_id = "performer-1",
                        name = "Performer One",
                        aliases = new[] { "Alias One" },
                        disambiguation = (string?)null,
                        sample_count = 3,
                        quality_score = 21.5,
                        image_url = "https://stashdb.org/images/one.jpg",
                    },
                    new
                    {
                        stashdb_id = "performer-2",
                        name = "Performer Two",
                        aliases = Array.Empty<string>(),
                        disambiguation = "Second performer",
                        sample_count = 1,
                        quality_score = 19.25,
                        image_url = "https://stashdb.org/images/two.jpg",
                    },
                ],
                [
                    1f, 2f, 3f, 4f,
                    5f, 6f, 7f, 8f,
                ],
                2,
                4);

            var store = new AiFaceReferencePackStore(Path.Combine(tempRoot, "reference"), new SaieArchiveReader());
            store.Attach(new TestExtensionStore());

            await using var source = File.OpenRead(archivePath);
            var stagedPath = await store.StageUploadAsync(source, Path.GetFileName(archivePath));
            var status = await store.ImportStagedAsync(stagedPath, Path.GetFileName(archivePath), new TestJobProgress());

            Assert.Equal("stashdb_server_20260323_202719", status.PackId);
            Assert.Equal(2, status.PerformerCount);
            Assert.True(File.Exists(status.ArchivePath));

            var activePack = await store.GetActivePackAsync();
            Assert.NotNull(activePack);
            Assert.Equal([1f, 2f, 3f, 4f], activePack!.GetCentroid(0).ToArray());

            var reloadedStatus = await store.GetStatusAsync();
            Assert.NotNull(reloadedStatus);
            Assert.Equal(status.ArchivePath, reloadedStatus!.ArchivePath);

            await store.ClearAsync();

            Assert.Null(await store.GetStatusAsync());
            Assert.Null(await store.GetActivePackAsync());
            Assert.False(File.Exists(status.ArchivePath));
        }
        finally
        {
            if (Directory.Exists(tempRoot))
                Directory.Delete(tempRoot, recursive: true);
        }
    }

    private static string CreateArchive(string root, string fileName, object manifest, IReadOnlyList<object> performers, IReadOnlyList<float> centroids, int rows, int columns)
    {
        var path = Path.Combine(root, fileName);

        using var file = File.Create(path);
        using var archive = new ZipArchive(file, ZipArchiveMode.Create, leaveOpen: false);

        var manifestEntry = archive.CreateEntry("manifest.json", CompressionLevel.Fastest);
        using (var writer = new StreamWriter(manifestEntry.Open(), Encoding.UTF8, leaveOpen: false))
        {
            writer.Write(JsonSerializer.Serialize(manifest));
        }

        var performersEntry = archive.CreateEntry("performers.jsonl", CompressionLevel.Fastest);
        using (var writer = new StreamWriter(performersEntry.Open(), Encoding.UTF8, leaveOpen: false))
        {
            foreach (var performer in performers)
                writer.WriteLine(JsonSerializer.Serialize(performer));
        }

        var centroidsEntry = archive.CreateEntry("centroids.npy", CompressionLevel.NoCompression);
        using (var stream = centroidsEntry.Open())
        {
            WriteNpy(stream, centroids, rows, columns);
        }

        return path;
    }

    private static void WriteNpy(Stream stream, IReadOnlyList<float> values, int rows, int columns)
    {
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);

        writer.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 0x01, 0x00 });

        var header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {columns}), }}";
        var preambleLength = 10;
        var paddingLength = 16 - ((preambleLength + header.Length + 1) % 16);
        if (paddingLength == 16)
            paddingLength = 0;

        var paddedHeader = header + new string(' ', paddingLength) + '\n';
        var headerBytes = Encoding.ASCII.GetBytes(paddedHeader);

        writer.Write((ushort)headerBytes.Length);
        writer.Write(headerBytes);

        foreach (var value in values)
            writer.Write(value);
    }

    private sealed class TestExtensionStore : IExtensionStore
    {
        private readonly Dictionary<string, string> _entries = new(StringComparer.OrdinalIgnoreCase);

        public Task<string?> GetAsync(string key, CancellationToken ct = default)
            => Task.FromResult<string?>(_entries.TryGetValue(key, out var value) ? value : null);

        public Task SetAsync(string key, string value, CancellationToken ct = default)
        {
            _entries[key] = value;
            return Task.CompletedTask;
        }

        public Task DeleteAsync(string key, CancellationToken ct = default)
        {
            _entries.Remove(key);
            return Task.CompletedTask;
        }

        public Task<Dictionary<string, string>> GetAllAsync(CancellationToken ct = default)
            => Task.FromResult(new Dictionary<string, string>(_entries, StringComparer.OrdinalIgnoreCase));
    }

    private sealed class TestJobProgress : CoreJobProgress
    {
        public void Report(double progress, string? subTask = null)
        {
        }
    }
}
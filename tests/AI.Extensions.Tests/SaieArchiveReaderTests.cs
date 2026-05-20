using System.IO.Compression;
using System.Text;
using System.Text.Json;

using AI.Faces;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class SaieArchiveReaderTests
{
    [Fact]
    public async Task ReadAsync_ParsesManifestIdentitiesAndCentroids()
    {
        var archivePath = CreateArchive(
            new
            {
                version = 1,
                embedder = "nsfw_ai_server:image_pipeline_face_embeddings_v1",
                embedding_dim = 4,
                pack_id = "sample-pack",
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

        try
        {
            var reader = new SaieArchiveReader();
            var pack = await reader.ReadAsync(archivePath);

            Assert.Equal("sample-pack", pack.Manifest.PackId);
            Assert.Equal(4, pack.Manifest.EmbeddingDim);
            Assert.Equal(2, pack.Manifest.PerformerCount);

            Assert.Equal(2, pack.Identities.Count);
            Assert.Equal("performer-1", pack.Identities[0].ExternalId);
            Assert.Equal("Performer One", pack.Identities[0].DisplayName);
            Assert.Equal(["Alias One"], pack.Identities[0].Aliases);
            Assert.Equal("Second performer", pack.Identities[1].Disambiguation);

            Assert.Equal([1f, 2f, 3f, 4f], pack.GetCentroid(0).ToArray());
            Assert.Equal([5f, 6f, 7f, 8f], pack.GetCentroid(1).ToArray());
        }
        finally
        {
            File.Delete(archivePath);
        }
    }

    [Fact]
    public async Task ReadAsync_ThrowsWhenPerformerRowsDoNotMatchCentroidRows()
    {
        var archivePath = CreateArchive(
            new
            {
                version = 1,
                embedder = "nsfw_ai_server:image_pipeline_face_embeddings_v1",
                embedding_dim = 2,
                pack_id = "bad-pack",
                source_endpoint = "https://stashdb.org/graphql",
                performer_count = 1,
                created_at = "2026-03-24T02:43:03Z",
            },
            [
                new
                {
                    stashdb_id = "performer-1",
                    name = "Performer One",
                    aliases = Array.Empty<string>(),
                    disambiguation = (string?)null,
                    sample_count = 1,
                    quality_score = 20.0,
                    image_url = "https://stashdb.org/images/one.jpg",
                },
            ],
            [
                1f, 2f,
                3f, 4f,
            ],
            2,
            2);

        try
        {
            var reader = new SaieArchiveReader();
            var error = await Assert.ThrowsAsync<InvalidOperationException>(() => reader.ReadAsync(archivePath));
            Assert.Contains("centroid array contains 2 rows", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(archivePath);
        }
    }

    private static string CreateArchive(object manifest, IReadOnlyList<object> performers, IReadOnlyList<float> centroids, int rows, int columns)
    {
        var path = Path.Combine(Path.GetTempPath(), $"saie-{Guid.NewGuid():N}.saie");

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
            {
                writer.WriteLine(JsonSerializer.Serialize(performer));
            }
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
}
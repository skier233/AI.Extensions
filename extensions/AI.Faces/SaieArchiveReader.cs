using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace AI.Faces;

internal sealed record SaieManifest(
    int Version,
    string Embedder,
    int EmbeddingDim,
    string PackId,
    string? SourceEndpoint,
    int PerformerCount,
    DateTimeOffset? CreatedAt);

internal sealed record SaieReferenceIdentity(
    int Ordinal,
    string ExternalId,
    string DisplayName,
    IReadOnlyList<string> Aliases,
    string? Disambiguation,
    int SampleCount,
    double QualityScore,
    string? ImageUrl);

internal sealed class SaieReferencePack(
    SaieManifest manifest,
    IReadOnlyList<SaieReferenceIdentity> identities,
    float[] centroids)
{
    public SaieManifest Manifest { get; } = manifest;

    public IReadOnlyList<SaieReferenceIdentity> Identities { get; } = identities;

    public float[] Centroids { get; } = centroids;

    public float[] CentroidNorms { get; } = ComputeCentroidNorms(manifest.EmbeddingDim, centroids);

    public ReadOnlySpan<float> GetCentroid(int ordinal)
    {
        if (ordinal < 0 || ordinal >= Identities.Count)
            throw new ArgumentOutOfRangeException(nameof(ordinal));

        var dimension = Manifest.EmbeddingDim;
        return Centroids.AsSpan(ordinal * dimension, dimension);
    }

    public float GetCentroidNorm(int ordinal)
    {
        if (ordinal < 0 || ordinal >= Identities.Count)
            throw new ArgumentOutOfRangeException(nameof(ordinal));

        return CentroidNorms[ordinal];
    }

    private static float[] ComputeCentroidNorms(int dimension, IReadOnlyList<float> centroids)
    {
        if (dimension <= 0 || centroids.Count == 0)
            return [];

        var norms = new float[centroids.Count / dimension];
        for (var row = 0; row < norms.Length; row++)
        {
            var sum = 0f;
            var offset = row * dimension;
            for (var index = 0; index < dimension; index++)
            {
                var value = centroids[offset + index];
                sum += value * value;
            }

            norms[row] = MathF.Sqrt(sum);
        }

        return norms;
    }
}

internal sealed class SaieArchiveReader
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        PropertyNameCaseInsensitive = true,
    };

    private static readonly Regex HeaderRegex = new(
        "'descr':\\s*'(?<descr>[^']+)'\\s*,\\s*'fortran_order':\\s*(?<fortran>True|False)\\s*,\\s*'shape':\\s*\\((?<shape>[^)]*)\\)",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    public async Task<SaieReferencePack> ReadAsync(string archivePath, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(archivePath);

        await using var stream = File.OpenRead(archivePath);
        return await ReadAsync(stream, cancellationToken);
    }

    public async Task<SaieReferencePack> ReadAsync(Stream archiveStream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(archiveStream);
        if (!archiveStream.CanRead)
            throw new InvalidOperationException("The supplied .saie stream is not readable.");

        using var archive = new ZipArchive(archiveStream, ZipArchiveMode.Read, leaveOpen: true);
        var manifest = await ReadManifestAsync(archive, cancellationToken);
        var identities = await ReadIdentitiesAsync(archive, cancellationToken);
        var centroids = ReadCentroids(archive, manifest, identities.Count);

        if (manifest.PerformerCount != identities.Count)
        {
            throw new InvalidOperationException(
                $"The .saie manifest declares {manifest.PerformerCount} performers but performers.jsonl contains {identities.Count} rows.");
        }

        return new SaieReferencePack(manifest, identities, centroids);
    }

    private static async Task<SaieManifest> ReadManifestAsync(ZipArchive archive, CancellationToken cancellationToken)
    {
        var entry = archive.GetEntry("manifest.json")
            ?? throw new InvalidOperationException("The .saie archive is missing manifest.json.");

        await using var stream = entry.Open();
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: false);
        var payload = await reader.ReadToEndAsync(cancellationToken);
        var manifest = JsonSerializer.Deserialize<SaieManifestPayload>(payload, JsonOptions)
            ?? throw new InvalidOperationException("The .saie manifest could not be parsed.");

        return new SaieManifest(
            manifest.Version,
            manifest.Embedder ?? throw new InvalidOperationException("The .saie manifest is missing embedder."),
            manifest.EmbeddingDim,
            manifest.PackId ?? throw new InvalidOperationException("The .saie manifest is missing pack_id."),
            manifest.SourceEndpoint,
            manifest.PerformerCount,
            manifest.CreatedAt);
    }

    private static async Task<IReadOnlyList<SaieReferenceIdentity>> ReadIdentitiesAsync(ZipArchive archive, CancellationToken cancellationToken)
    {
        var entry = archive.GetEntry("performers.jsonl")
            ?? throw new InvalidOperationException("The .saie archive is missing performers.jsonl.");

        var identities = new List<SaieReferenceIdentity>();
        await using var stream = entry.Open();
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: false);

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var line = await reader.ReadLineAsync(cancellationToken);
            if (line is null)
                break;

            if (string.IsNullOrWhiteSpace(line))
                continue;

            var payload = JsonSerializer.Deserialize<SaieIdentityPayload>(line, JsonOptions)
                ?? throw new InvalidOperationException("A performers.jsonl row could not be parsed.");

            identities.Add(new SaieReferenceIdentity(
                identities.Count,
                payload.StashDbId ?? throw new InvalidOperationException("A performers.jsonl row is missing stashdb_id."),
                payload.Name ?? throw new InvalidOperationException("A performers.jsonl row is missing name."),
                payload.Aliases ?? [],
                payload.Disambiguation,
                payload.SampleCount,
                payload.QualityScore,
                payload.ImageUrl));
        }

        return identities;
    }

    private static float[] ReadCentroids(ZipArchive archive, SaieManifest manifest, int identityCount)
    {
        var entry = archive.GetEntry("centroids.npy")
            ?? throw new InvalidOperationException("The .saie archive is missing centroids.npy.");

        using var stream = entry.Open();
        var header = ReadNpyHeader(stream);

        if (!string.Equals(header.Descriptor, "<f4", StringComparison.Ordinal)
            && !string.Equals(header.Descriptor, "|f4", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Unsupported .npy descriptor '{header.Descriptor}'. Expected float32 centroids.");
        }

        if (header.FortranOrder)
            throw new InvalidOperationException("Fortran-ordered .npy arrays are not supported for .saie centroids.");

        if (header.Shape.Length != 2)
            throw new InvalidOperationException("The .saie centroid array must be a 2D matrix.");

        if (header.Shape[0] != identityCount)
        {
            throw new InvalidOperationException(
                $"The .saie centroid array contains {header.Shape[0]} rows but performers.jsonl contains {identityCount} rows.");
        }

        if (header.Shape[1] != manifest.EmbeddingDim)
        {
            throw new InvalidOperationException(
                $"The .saie centroid array uses dimension {header.Shape[1]} but manifest.json declares {manifest.EmbeddingDim}.");
        }

        var valueCount = checked(header.Shape[0] * header.Shape[1]);
        var values = new float[valueCount];
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        for (var index = 0; index < values.Length; index++)
        {
            values[index] = reader.ReadSingle();
        }

        return values;
    }

    private static NpyHeader ReadNpyHeader(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);

        var magic = reader.ReadBytes(6);
        var expectedMagic = new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };
        if (!magic.SequenceEqual(expectedMagic))
            throw new InvalidOperationException("The .saie centroid payload is not a valid .npy stream.");

        var majorVersion = reader.ReadByte();
        _ = reader.ReadByte();

        var headerLength = majorVersion switch
        {
            1 => reader.ReadUInt16(),
            2 => checked((int)reader.ReadUInt32()),
            _ => throw new InvalidOperationException($"Unsupported .npy format version {majorVersion}."),
        };

        var headerText = Encoding.ASCII.GetString(reader.ReadBytes(headerLength)).Trim();
        var match = HeaderRegex.Match(headerText);
        if (!match.Success)
            throw new InvalidOperationException("The .npy header could not be parsed.");

        var descriptor = match.Groups["descr"].Value;
        var fortranOrder = string.Equals(match.Groups["fortran"].Value, "True", StringComparison.Ordinal);
        var shapeParts = match.Groups["shape"].Value
            .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        var shape = shapeParts
            .Select(part => int.Parse(part, NumberStyles.Integer, CultureInfo.InvariantCulture))
            .ToArray();

        return new NpyHeader(descriptor, fortranOrder, shape);
    }

    private sealed record NpyHeader(string Descriptor, bool FortranOrder, int[] Shape);

    private sealed class SaieManifestPayload
    {
        [JsonPropertyName("version")]
        public int Version { get; set; }

        [JsonPropertyName("embedder")]
        public string? Embedder { get; set; }

        [JsonPropertyName("embedding_dim")]
        public int EmbeddingDim { get; set; }

        [JsonPropertyName("pack_id")]
        public string? PackId { get; set; }

        [JsonPropertyName("source_endpoint")]
        public string? SourceEndpoint { get; set; }

        [JsonPropertyName("performer_count")]
        public int PerformerCount { get; set; }

        [JsonPropertyName("created_at")]
        public DateTimeOffset? CreatedAt { get; set; }
    }

    private sealed class SaieIdentityPayload
    {
        [JsonPropertyName("stashdb_id")]
        public string? StashDbId { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("aliases")]
        public List<string>? Aliases { get; set; }

        [JsonPropertyName("disambiguation")]
        public string? Disambiguation { get; set; }

        [JsonPropertyName("sample_count")]
        public int SampleCount { get; set; }

        [JsonPropertyName("quality_score")]
        public double QualityScore { get; set; }

        [JsonPropertyName("image_url")]
        public string? ImageUrl { get; set; }
    }
}
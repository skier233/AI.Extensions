using System.Text.Json;

using Cove.Core.Interfaces;
using Cove.Plugins;

using CoreJobProgress = Cove.Core.Interfaces.IJobProgress;

namespace AI.Faces;

internal sealed record AiFaceReferencePackStatus(
    string PackId,
    string FileName,
    string ArchivePath,
    string Embedder,
    int EmbeddingDim,
    int PerformerCount,
    string? SourceEndpoint,
    DateTimeOffset? SourceCreatedAt,
    DateTimeOffset ImportedAt);

internal sealed class AiFaceReferencePackStore(string referenceRootPath, SaieArchiveReader archiveReader)
{
    private const string StateStoreKey = "reference-pack-state";

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false,
    };

    private readonly string _referenceRootPath = referenceRootPath;
    private readonly SaieArchiveReader _archiveReader = archiveReader;
    private readonly SemaphoreSlim _gate = new(1, 1);

    private IExtensionStore? _store;
    private SaieReferencePack? _cachedPack;
    private string? _cachedArchivePath;
    private DateTime _cachedArchiveWriteUtc;

    public void Attach(IExtensionStore store)
    {
        _store = store;
    }

    public async Task<string> StageUploadAsync(Stream source, string fileName, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);

        Directory.CreateDirectory(GetIncomingDirectory());
        var stagedPath = Path.Combine(GetIncomingDirectory(), $"{Guid.NewGuid():N}{Path.GetExtension(fileName)}");

        await using var destination = File.Create(stagedPath);
        await source.CopyToAsync(destination, ct);
        await destination.FlushAsync(ct);

        return stagedPath;
    }

    public async Task<AiFaceReferencePackStatus> ImportStagedAsync(
        string stagedArchivePath,
        string originalFileName,
        CoreJobProgress? progress = null,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stagedArchivePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(originalFileName);
        EnsureStoreAttached();

        await _gate.WaitAsync(ct);
        try
        {
            progress?.Report(0.05, "Validating .saie archive...");
            var pack = await _archiveReader.ReadAsync(stagedArchivePath, ct);

            Directory.CreateDirectory(_referenceRootPath);
            var targetFileName = $"{SanitizeFileName(pack.Manifest.PackId)}.saie";
            var targetPath = Path.Combine(_referenceRootPath, targetFileName);

            progress?.Report(0.4, "Persisting reference pack...");
            if (File.Exists(targetPath))
                File.Delete(targetPath);

            File.Move(stagedArchivePath, targetPath);

            var current = await GetStatusCoreAsync(ct);
            if (current is not null
                && !string.Equals(current.ArchivePath, targetPath, StringComparison.OrdinalIgnoreCase)
                && File.Exists(current.ArchivePath))
            {
                File.Delete(current.ArchivePath);
            }

            var status = new AiFaceReferencePackStatus(
                pack.Manifest.PackId,
                Path.GetFileName(originalFileName),
                targetPath,
                pack.Manifest.Embedder,
                pack.Manifest.EmbeddingDim,
                pack.Manifest.PerformerCount,
                pack.Manifest.SourceEndpoint,
                pack.Manifest.CreatedAt,
                DateTimeOffset.UtcNow);

            await _store!.SetAsync(StateStoreKey, JsonSerializer.Serialize(status, JsonOptions), ct);

            _cachedPack = pack;
            _cachedArchivePath = targetPath;
            _cachedArchiveWriteUtc = File.GetLastWriteTimeUtc(targetPath);

            progress?.Report(1.0, $"Imported {pack.Manifest.PerformerCount} reference identities.");
            return status;
        }
        finally
        {
            _gate.Release();
            TryDeleteFile(stagedArchivePath);
        }
    }

    public async Task<AiFaceReferencePackStatus?> GetStatusAsync(CancellationToken ct = default)
    {
        EnsureStoreAttached();
        return await GetStatusCoreAsync(ct);
    }

    public async Task<SaieReferencePack?> GetActivePackAsync(CancellationToken ct = default)
    {
        EnsureStoreAttached();

        await _gate.WaitAsync(ct);
        try
        {
            var status = await GetStatusCoreAsync(ct);
            if (status is null || !File.Exists(status.ArchivePath))
                return null;

            var lastWriteUtc = File.GetLastWriteTimeUtc(status.ArchivePath);
            if (_cachedPack is not null
                && string.Equals(_cachedArchivePath, status.ArchivePath, StringComparison.OrdinalIgnoreCase)
                && _cachedArchiveWriteUtc == lastWriteUtc)
            {
                return _cachedPack;
            }

            var pack = await _archiveReader.ReadAsync(status.ArchivePath, ct);
            _cachedPack = pack;
            _cachedArchivePath = status.ArchivePath;
            _cachedArchiveWriteUtc = lastWriteUtc;
            return pack;
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task ClearAsync(CancellationToken ct = default)
    {
        EnsureStoreAttached();

        await _gate.WaitAsync(ct);
        try
        {
            var status = await GetStatusCoreAsync(ct);
            if (status is not null && File.Exists(status.ArchivePath))
                File.Delete(status.ArchivePath);

            if (_store is not null)
                await _store.DeleteAsync(StateStoreKey, ct);

            _cachedPack = null;
            _cachedArchivePath = null;
            _cachedArchiveWriteUtc = default;
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task<AiFaceReferencePackStatus?> GetStatusCoreAsync(CancellationToken ct)
    {
        if (_store is null)
            return null;

        var payload = await _store.GetAsync(StateStoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
            return null;

        return JsonSerializer.Deserialize<AiFaceReferencePackStatus>(payload, JsonOptions);
    }

    private string GetIncomingDirectory()
        => Path.Combine(_referenceRootPath, ".incoming");

    private void EnsureStoreAttached()
    {
        if (_store is null)
            throw new InvalidOperationException("AI.Faces reference pack store is not attached to the extension store.");
    }

    private static string SanitizeFileName(string value)
    {
        var invalid = Path.GetInvalidFileNameChars();
        return string.Concat(value.Select(character => invalid.Contains(character) ? '_' : character));
    }

    private static void TryDeleteFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch
        {
        }
    }
}
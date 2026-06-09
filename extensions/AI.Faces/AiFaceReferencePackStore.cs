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

// Stores one or more imported .saie reference packs. Each pack is a per-site performer reference set;
// multiple packs (e.g. from different metadata sites) can be active simultaneously and are all matched
// against during suggestion. Packs are keyed by their manifest PackId and persisted both as files
// under <referenceRootPath> and as a JSON index in the extension store.
//
// The reference root is deliberately located outside the extension's own install/version directory so
// an extension update (which replaces that directory) never deletes imported packs. On first run any
// pack files / index written by the previous single-pack layout are migrated forward.
internal sealed class AiFaceReferencePackStore(
    string referenceRootPath,
    SaieArchiveReader archiveReader,
    string? legacyReferenceRootPath = null)
{
    private const string StateStoreKey = "reference-packs-state";
    private const string LegacyStateStoreKey = "reference-pack-state";

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false,
    };

    private readonly string _referenceRootPath = referenceRootPath;
    private readonly string? _legacyReferenceRootPath = legacyReferenceRootPath;
    private readonly SaieArchiveReader _archiveReader = archiveReader;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Dictionary<string, CachedPack> _cache = new(StringComparer.OrdinalIgnoreCase);

    private IExtensionStore? _store;
    private bool _migrated;

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
            await EnsureMigratedAsync(ct);

            progress?.Report(0.05, "Validating .saie archive...");
            var pack = await _archiveReader.ReadAsync(stagedArchivePath, ct);

            Directory.CreateDirectory(_referenceRootPath);
            var targetFileName = $"{SanitizeFileName(pack.Manifest.PackId)}.saie";
            var targetPath = Path.Combine(_referenceRootPath, targetFileName);

            progress?.Report(0.4, "Persisting reference pack...");

            var statuses = await GetStatusesCoreAsync(ct);
            // Replacing a pack that already exists with the same id: drop its old file unless it maps
            // to the same path we are about to write.
            var existing = statuses.FirstOrDefault(status => string.Equals(status.PackId, pack.Manifest.PackId, StringComparison.OrdinalIgnoreCase));
            if (existing is not null
                && !string.Equals(existing.ArchivePath, targetPath, StringComparison.OrdinalIgnoreCase)
                && File.Exists(existing.ArchivePath))
            {
                File.Delete(existing.ArchivePath);
            }

            if (File.Exists(targetPath))
                File.Delete(targetPath);

            File.Move(stagedArchivePath, targetPath);

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

            var updated = statuses
                .Where(item => !string.Equals(item.PackId, status.PackId, StringComparison.OrdinalIgnoreCase))
                .Append(status)
                .OrderBy(item => item.PackId, StringComparer.OrdinalIgnoreCase)
                .ToList();
            await SaveStatusesCoreAsync(updated, ct);

            _cache[targetPath] = new CachedPack(pack, File.GetLastWriteTimeUtc(targetPath));

            progress?.Report(1.0, $"Imported {pack.Manifest.PerformerCount} reference identities from {pack.Manifest.PackId}.");
            return status;
        }
        finally
        {
            _gate.Release();
            TryDeleteFile(stagedArchivePath);
        }
    }

    // Convenience single-pack accessors for callers/tests that predate multi-pack support; they return
    // the first pack in deterministic order.
    public async Task<AiFaceReferencePackStatus?> GetStatusAsync(CancellationToken ct = default)
        => (await GetStatusesAsync(ct)).FirstOrDefault();

    public async Task<SaieReferencePack?> GetActivePackAsync(CancellationToken ct = default)
        => (await GetActivePacksAsync(ct)).FirstOrDefault();

    public async Task<IReadOnlyList<AiFaceReferencePackStatus>> GetStatusesAsync(CancellationToken ct = default)
    {
        EnsureStoreAttached();
        await _gate.WaitAsync(ct);
        try
        {
            await EnsureMigratedAsync(ct);
            return await GetStatusesCoreAsync(ct);
        }
        finally
        {
            _gate.Release();
        }
    }

    // All currently-importable packs, in a deterministic order (by PackId). The list index is the
    // stable "pack index" used to encode reference suggestion ids, so suggester and decision handler
    // observe the same ordering.
    public async Task<IReadOnlyList<SaieReferencePack>> GetActivePacksAsync(CancellationToken ct = default)
    {
        EnsureStoreAttached();

        await _gate.WaitAsync(ct);
        try
        {
            await EnsureMigratedAsync(ct);

            var statuses = await GetStatusesCoreAsync(ct);
            var packs = new List<SaieReferencePack>(statuses.Count);
            foreach (var status in statuses)
            {
                if (!File.Exists(status.ArchivePath))
                    continue;

                var lastWriteUtc = File.GetLastWriteTimeUtc(status.ArchivePath);
                if (_cache.TryGetValue(status.ArchivePath, out var cached) && cached.WriteUtc == lastWriteUtc)
                {
                    packs.Add(cached.Pack);
                    continue;
                }

                var pack = await _archiveReader.ReadAsync(status.ArchivePath, ct);
                _cache[status.ArchivePath] = new CachedPack(pack, lastWriteUtc);
                packs.Add(pack);
            }

            return packs;
        }
        finally
        {
            _gate.Release();
        }
    }

    // Removes a single pack by id, or every pack when packId is null/blank.
    public async Task ClearAsync(string? packId = null, CancellationToken ct = default)
    {
        EnsureStoreAttached();

        await _gate.WaitAsync(ct);
        try
        {
            await EnsureMigratedAsync(ct);

            var statuses = await GetStatusesCoreAsync(ct);
            var removing = string.IsNullOrWhiteSpace(packId)
                ? statuses
                : statuses.Where(status => string.Equals(status.PackId, packId, StringComparison.OrdinalIgnoreCase)).ToList();

            foreach (var status in removing)
            {
                if (File.Exists(status.ArchivePath))
                    File.Delete(status.ArchivePath);
                _cache.Remove(status.ArchivePath);
            }

            var remaining = statuses
                .Where(status => !removing.Contains(status))
                .ToList();

            if (remaining.Count == 0 && _store is not null)
                await _store.DeleteAsync(StateStoreKey, ct);
            else
                await SaveStatusesCoreAsync(remaining, ct);
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task<IReadOnlyList<AiFaceReferencePackStatus>> GetStatusesCoreAsync(CancellationToken ct)
    {
        if (_store is null)
            return [];

        var payload = await _store.GetAsync(StateStoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
            return [];

        var statuses = JsonSerializer.Deserialize<List<AiFaceReferencePackStatus>>(payload, JsonOptions);
        return statuses is null
            ? []
            : statuses.OrderBy(status => status.PackId, StringComparer.OrdinalIgnoreCase).ToList();
    }

    private async Task SaveStatusesCoreAsync(IReadOnlyList<AiFaceReferencePackStatus> statuses, CancellationToken ct)
    {
        if (_store is null)
            return;

        await _store.SetAsync(StateStoreKey, JsonSerializer.Serialize(statuses, JsonOptions), ct);
    }

    // Forward-migrate the previous single-pack layout: pull the legacy state-store entry into the new
    // index and relocate any pack file that still lives under the old (update-vulnerable) directory.
    private async Task EnsureMigratedAsync(CancellationToken ct)
    {
        if (_migrated || _store is null)
        {
            _migrated = true;
            return;
        }

        var current = await _store.GetAsync(StateStoreKey, ct);
        var legacy = await _store.GetAsync(LegacyStateStoreKey, ct);

        if (string.IsNullOrWhiteSpace(current) && !string.IsNullOrWhiteSpace(legacy))
        {
            var legacyStatus = JsonSerializer.Deserialize<AiFaceReferencePackStatus>(legacy, JsonOptions);
            if (legacyStatus is not null)
            {
                var migratedStatus = legacyStatus;
                if (File.Exists(legacyStatus.ArchivePath)
                    && !PathIsUnder(legacyStatus.ArchivePath, _referenceRootPath))
                {
                    Directory.CreateDirectory(_referenceRootPath);
                    var targetPath = Path.Combine(_referenceRootPath, $"{SanitizeFileName(legacyStatus.PackId)}.saie");
                    try
                    {
                        if (!File.Exists(targetPath))
                            File.Copy(legacyStatus.ArchivePath, targetPath);
                        migratedStatus = legacyStatus with { ArchivePath = targetPath };
                    }
                    catch
                    {
                        // Keep the legacy path if the relocation fails; it will still load until the
                        // next update removes it.
                    }
                }

                await SaveStatusesCoreAsync([migratedStatus], ct);
            }
        }

        if (!string.IsNullOrWhiteSpace(legacy))
            await _store.DeleteAsync(LegacyStateStoreKey, ct);

        _migrated = true;
    }

    private string GetIncomingDirectory()
        => Path.Combine(_referenceRootPath, ".incoming");

    private void EnsureStoreAttached()
    {
        if (_store is null)
            throw new InvalidOperationException("AI.Faces reference pack store is not attached to the extension store.");
    }

    private static bool PathIsUnder(string path, string root)
    {
        var fullPath = Path.GetFullPath(path);
        var fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return fullPath.StartsWith(fullRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
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

    private readonly record struct CachedPack(SaieReferencePack Pack, DateTime WriteUtc);
}

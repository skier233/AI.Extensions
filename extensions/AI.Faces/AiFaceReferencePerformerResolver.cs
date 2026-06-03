using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Faces;

internal sealed record AiFaceReferencePerformerMatch(
    int PerformerId,
    string PerformerName,
    DateTime? PerformerUpdatedAt,
    bool LocalPerformerHasImage,
    bool LocalPerformerIsLocalOnly);

internal sealed class AiFaceReferencePerformerResolver(IPerformerRepository performerRepository)
{
    private readonly IPerformerRepository _performerRepository = performerRepository;

    public async Task<IReadOnlyDictionary<string, AiFaceReferencePerformerMatch>> ResolveAsync(
        IReadOnlyCollection<SaieReferenceIdentity> identities,
        string? sourceEndpoint,
        CancellationToken ct = default)
    {
        if (identities.Count == 0)
            return new Dictionary<string, AiFaceReferencePerformerMatch>(StringComparer.OrdinalIgnoreCase);

        var externalIds = identities
            .Select(identity => identity.ExternalId)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var searchNames = identities
            .SelectMany(GetSearchNames)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var performers = await _performerRepository.FindByNamesOrRemoteIdsAsync(
            searchNames, sourceEndpoint, externalIds, ct);

        var matches = new Dictionary<string, AiFaceReferencePerformerMatch>(StringComparer.OrdinalIgnoreCase);
        foreach (var identity in identities)
        {
            var performer = SelectMatch(performers, identity, sourceEndpoint);
            if (performer is null)
                continue;

            matches[identity.ExternalId] = new AiFaceReferencePerformerMatch(
                performer.Id,
                performer.Name,
                performer.UpdatedAt,
                !string.IsNullOrWhiteSpace(performer.ImageBlobId),
                performer.RemoteIds.Count == 0);
        }

        return matches;
    }

    public async Task<Performer> FindOrCreateAsync(
        SaieReferenceIdentity identity,
        string? sourceEndpoint,
        CancellationToken ct = default)
    {
        if (!string.IsNullOrWhiteSpace(sourceEndpoint))
        {
            var existingByRemoteId = await _performerRepository.FindByRemoteIdAsync(
                sourceEndpoint, identity.ExternalId, ct);
            if (existingByRemoteId is not null)
                return existingByRemoteId;
        }

        var searchNames = GetSearchNames(identity).ToArray();
        var nameMatches = await _performerRepository.FindByNamesOrRemoteIdsAsync(
            searchNames, null, [], ct);
        if (nameMatches.Count == 1)
            return nameMatches[0];

        var performer = new Performer
        {
            Name = identity.DisplayName,
            Disambiguation = identity.Disambiguation,
        };

        foreach (var alias in identity.Aliases
                     .Where(alias => !string.IsNullOrWhiteSpace(alias))
                     .Select(alias => alias.Trim())
                     .Where(alias => !string.Equals(alias, identity.DisplayName, StringComparison.OrdinalIgnoreCase))
                     .Distinct(StringComparer.OrdinalIgnoreCase))
        {
            performer.Aliases.Add(new PerformerAlias { Alias = alias });
        }

        if (!string.IsNullOrWhiteSpace(sourceEndpoint))
        {
            performer.RemoteIds.Add(new PerformerRemoteId
            {
                Endpoint = sourceEndpoint,
                RemoteId = identity.ExternalId,
            });
        }

        return await _performerRepository.AddAsync(performer, ct);
    }

    private static Performer? SelectMatch(IReadOnlyCollection<Performer> performers, SaieReferenceIdentity identity, string? sourceEndpoint)
    {
        if (!string.IsNullOrWhiteSpace(sourceEndpoint))
        {
            var remoteMatches = performers
                .Where(performer => performer.RemoteIds.Any(remoteId => remoteId.Endpoint == sourceEndpoint && remoteId.RemoteId == identity.ExternalId))
                .ToArray();
            if (remoteMatches.Length == 1)
                return remoteMatches[0];
        }

        var searchNames = GetSearchNames(identity).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var nameMatches = performers
            .Where(performer => searchNames.Contains(performer.Name) || performer.Aliases.Any(alias => searchNames.Contains(alias.Alias)))
            .ToArray();

        return nameMatches.Length == 1 ? nameMatches[0] : null;
    }

    private static IEnumerable<string> GetSearchNames(SaieReferenceIdentity identity)
    {
        if (!string.IsNullOrWhiteSpace(identity.DisplayName))
            yield return identity.DisplayName.Trim();

        foreach (var alias in identity.Aliases.Where(alias => !string.IsNullOrWhiteSpace(alias)).Select(alias => alias.Trim()))
            yield return alias;
    }
}

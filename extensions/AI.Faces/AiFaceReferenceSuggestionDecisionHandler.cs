using Cove.Core.Auth;
using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Microsoft.AspNetCore.Http;

namespace AI.Faces;

internal sealed class AiFaceReferenceSuggestionDecisionHandler(
    IFaceRepository faceRepository,
    AiFaceReferencePackStore packStore,
    AiFaceReferencePerformerResolver performerResolver,
    AiFaceReferenceSuggestionDecisionStore decisionStore,
    IFacePerformerPropagationService facePerformerPropagationService,
    IPerformerRepository performerRepository,
    IPerformerMergeService performerMergeService,
    ICurrentPrincipalAccessor principalAccessor) : IFaceSuggestionDecisionHandler
{
    public async Task<FaceSuggestionDecisionOutcome> TryHandleAsync(FaceSuggestionDecisionRequest request, CancellationToken cancellationToken = default)
    {
        if (string.Equals(request.Decision, FaceSuggestionDecisionValues.Merge, StringComparison.OrdinalIgnoreCase))
            return await TryHandleMergeAsync(request, cancellationToken);

        if (request.PerformerId >= 0)
            return FaceSuggestionDecisionOutcome.NotHandled;

        var principal = principalAccessor.Current;
        if (principal?.Has(AiFacesExtension.ApplyReferencePermission) != true)
            return FaceSuggestionDecisionOutcome.Failure($"Permission '{AiFacesExtension.ApplyReferencePermission}' is required.", principal is null || principal.Kind == PrincipalKind.Anonymous ? StatusCodes.Status401Unauthorized : StatusCodes.Status403Forbidden);

        var packs = await packStore.GetActivePacksAsync(cancellationToken);
        if (packs.Count == 0 || !AiFaceReferenceSuggestionIds.TryResolve(packs, request.PerformerId, out var pack, out var identity) || pack is null || identity is null)
            return FaceSuggestionDecisionOutcome.Failure("referenceSuggestionId did not resolve to an imported reference identity.");

        var sourceEndpoint = pack.Manifest.SourceEndpoint;

        if (request.Decision == FaceSuggestionDecisionValues.Reject)
        {
            var faceExists = await faceRepository.FaceExistsAsync(request.FaceId, cancellationToken);
            if (!faceExists)
                return FaceSuggestionDecisionOutcome.Failure("Face was not found.", StatusCodes.Status404NotFound);

            await decisionStore.RejectAsync(request.FaceId, identity.ExternalId, cancellationToken);
            return FaceSuggestionDecisionOutcome.Success;
        }

        if (request.Decision != FaceSuggestionDecisionValues.Accept)
            return FaceSuggestionDecisionOutcome.NotHandled;

        var face = await faceRepository.GetFaceAsync(request.FaceId, tracking: true, cancellationToken);
        if (face is null)
            return FaceSuggestionDecisionOutcome.Failure("Face was not found.", StatusCodes.Status404NotFound);

        var performer = await performerResolver.FindOrCreateAsync(identity, sourceEndpoint, cancellationToken);
        await decisionStore.ClearAsync(request.FaceId, identity.ExternalId, cancellationToken);
        await facePerformerPropagationService.ApplyLinkChangeAsync(request.FaceId, face.PerformerId, performer.Id, cancellationToken);
        face.PerformerId = performer.Id;
        await faceRepository.SaveChangesAsync(cancellationToken);

        return FaceSuggestionDecisionOutcome.Success;
    }

    // Folds two or more competing matches for one face into a single performer. The primary
    // (request.PerformerId) becomes the surviving performer; each secondary is resolved to a local
    // performer (creating one from its reference identity when needed) and merged in, so a person known
    // under different names across reference sources ends up as one performer with the others' names as
    // aliases and their site ids/links retained. The face is then linked to the primary.
    private async Task<FaceSuggestionDecisionOutcome> TryHandleMergeAsync(FaceSuggestionDecisionRequest request, CancellationToken cancellationToken)
    {
        var principal = principalAccessor.Current;
        if (principal?.Has(AiFacesExtension.ApplyReferencePermission) != true)
            return FaceSuggestionDecisionOutcome.Failure($"Permission '{AiFacesExtension.ApplyReferencePermission}' is required.", principal is null || principal.Kind == PrincipalKind.Anonymous ? StatusCodes.Status401Unauthorized : StatusCodes.Status403Forbidden);

        var secondaryIds = (request.SecondaryPerformerIds ?? []).Where(id => id != request.PerformerId).Distinct().ToArray();
        if (secondaryIds.Length == 0)
            return FaceSuggestionDecisionOutcome.Failure("Merge requires at least one other match to combine with.");

        var face = await faceRepository.GetFaceAsync(request.FaceId, tracking: true, cancellationToken);
        if (face is null)
            return FaceSuggestionDecisionOutcome.Failure("Face was not found.", StatusCodes.Status404NotFound);

        var packs = await packStore.GetActivePacksAsync(cancellationToken);

        var resolvedPrimary = await ResolveSuggestionPerformerAsync(request.PerformerId, packs, cancellationToken);
        if (resolvedPrimary is null)
            return FaceSuggestionDecisionOutcome.Failure("The primary match could not be resolved to a performer.");
        var (primaryId, primaryIdentity) = resolvedPrimary.Value;

        var sourceIds = new List<int>();
        foreach (var secondaryId in secondaryIds)
        {
            var resolvedSecondary = await ResolveSuggestionPerformerAsync(secondaryId, packs, cancellationToken);
            if (resolvedSecondary is null)
                continue;

            var (secondaryPerformerId, secondaryIdentity) = resolvedSecondary.Value;
            if (secondaryPerformerId != primaryId && !sourceIds.Contains(secondaryPerformerId))
                sourceIds.Add(secondaryPerformerId);

            if (secondaryIdentity is not null)
                await decisionStore.ClearAsync(request.FaceId, secondaryIdentity.ExternalId, cancellationToken);
        }

        if (sourceIds.Count > 0)
        {
            var merged = await performerMergeService.MergeAsync(primaryId, sourceIds, cancellationToken);
            if (merged is null)
                return FaceSuggestionDecisionOutcome.Failure("The primary performer could not be merged.");
        }

        if (primaryIdentity is not null)
            await decisionStore.ClearAsync(request.FaceId, primaryIdentity.ExternalId, cancellationToken);

        await facePerformerPropagationService.ApplyLinkChangeAsync(request.FaceId, face.PerformerId, primaryId, cancellationToken);
        face.PerformerId = primaryId;
        await faceRepository.SaveChangesAsync(cancellationToken);

        return FaceSuggestionDecisionOutcome.Success;
    }

    // Resolves a suggestion id to a concrete local performer. Positive ids are existing performers;
    // negative ids are reference identities, which are found-or-created (and possibly hydrated) via the
    // resolver. Returns the performer id and, for reference ids, the originating identity so any stale
    // suggestion decisions can be cleared.
    private async Task<(int PerformerId, SaieReferenceIdentity? Identity)?> ResolveSuggestionPerformerAsync(
        int suggestionId,
        IReadOnlyList<SaieReferencePack> packs,
        CancellationToken cancellationToken)
    {
        if (suggestionId >= 0)
        {
            var existing = await performerRepository.GetByIdAsync(suggestionId, cancellationToken);
            if (existing is null)
                return null;
            return (existing.Id, (SaieReferenceIdentity?)null);
        }

        if (packs.Count == 0 || !AiFaceReferenceSuggestionIds.TryResolve(packs, suggestionId, out var pack, out var identity) || pack is null || identity is null)
            return null;

        var performer = await performerResolver.FindOrCreateAsync(identity, pack.Manifest.SourceEndpoint, cancellationToken);
        return (performer.Id, identity);
    }
}

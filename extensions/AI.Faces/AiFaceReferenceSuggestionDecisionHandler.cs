using Cove.Core.Auth;
using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Cove.Data;
using Cove.Data.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;

namespace AI.Faces;

internal sealed class AiFaceReferenceSuggestionDecisionHandler(
    CoveContext db,
    AiFaceReferencePackStore packStore,
    AiFaceReferencePerformerResolver performerResolver,
    AiFaceReferenceSuggestionDecisionStore decisionStore,
    FacePerformerPropagationService facePerformerPropagationService,
    ICurrentPrincipalAccessor principalAccessor) : IFaceSuggestionDecisionHandler
{
    public async Task<FaceSuggestionDecisionOutcome> TryHandleAsync(FaceSuggestionDecisionRequest request, CancellationToken cancellationToken = default)
    {
        if (request.PerformerId >= 0)
            return FaceSuggestionDecisionOutcome.NotHandled;

        var principal = principalAccessor.Current;
        if (principal?.Has(AiFacesExtension.ApplyReferencePermission) != true)
            return FaceSuggestionDecisionOutcome.Failure($"Permission '{AiFacesExtension.ApplyReferencePermission}' is required.", principal is null || principal.Kind == PrincipalKind.Anonymous ? StatusCodes.Status401Unauthorized : StatusCodes.Status403Forbidden);

        var pack = await packStore.GetActivePackAsync(cancellationToken);
        if (pack is null || !AiFaceReferenceSuggestionIds.TryResolve(pack.Identities, request.PerformerId, out var identity) || identity is null)
            return FaceSuggestionDecisionOutcome.Failure("referenceSuggestionId did not resolve to an imported reference identity.");

        if (request.Decision == FaceSuggestionDecisionValues.Reject)
        {
            var faceExists = await db.Faces.AsNoTracking().AnyAsync(face => face.Id == request.FaceId, cancellationToken);
            if (!faceExists)
                return FaceSuggestionDecisionOutcome.Failure("Face was not found.", StatusCodes.Status404NotFound);

            await decisionStore.RejectAsync(request.FaceId, identity.ExternalId, cancellationToken);
            return FaceSuggestionDecisionOutcome.Success;
        }

        if (request.Decision != FaceSuggestionDecisionValues.Accept)
            return FaceSuggestionDecisionOutcome.NotHandled;

        var face = await db.Faces.FirstOrDefaultAsync(item => item.Id == request.FaceId, cancellationToken);
        if (face is null)
            return FaceSuggestionDecisionOutcome.Failure("Face was not found.", StatusCodes.Status404NotFound);

        var status = await packStore.GetStatusAsync(cancellationToken);
        if (status is null)
            return FaceSuggestionDecisionOutcome.Failure("Import a reference .saie pack before linking reference identities.");

        var performer = await performerResolver.FindOrCreateAsync(identity, status.SourceEndpoint, cancellationToken);
        await decisionStore.ClearAsync(request.FaceId, identity.ExternalId, cancellationToken);
        await facePerformerPropagationService.ApplyLinkChangeAsync(request.FaceId, face.PerformerId, performer.Id, cancellationToken);
        face.PerformerId = performer.Id;
        await db.SaveChangesAsync(cancellationToken);

        return FaceSuggestionDecisionOutcome.Success;
    }
}
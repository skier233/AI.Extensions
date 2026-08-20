using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Faces;

internal sealed class AiFacesDeleteParticipant(
    IFaceIdentityStore store,
    AiFaceIdentityExclusionStore? exclusionStore = null) : IFaceLifecycleParticipant
{
    private const string FaceSourceKey = "ext:ai.faces";

    private readonly IFaceIdentityStore _store = store;
    private readonly AiFaceIdentityExclusionStore? _exclusionStore = exclusionStore;

    public async Task OnDeletingAsync(Face face, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(face.PrimarySourceKey))
        {
            return;
        }

        await _store.DeleteByFaceKeyAsync(face.PrimarySourceKey, cancellationToken);
        if (_exclusionStore is not null)
        {
            await _exclusionStore.RemoveAsync(face.PrimarySourceKey, cancellationToken);
        }
    }

    public async Task OnFacesPurgedAsync(FacePurgeScope scope, CancellationToken cancellationToken = default)
    {
        // An "entire source" clear of this extension's (or all) face data must also drop provisional
        // identities, which have no Cove Face row and so are never reached by OnDeletingAsync. Narrower
        // purges leave the working identity graph intact (the per-face path handles promoted identities).
        var clearsThisSource = string.IsNullOrWhiteSpace(scope.SourceKey)
            || string.Equals(scope.SourceKey, FaceSourceKey, StringComparison.OrdinalIgnoreCase);
        if (!scope.IsEntireSource || !clearsThisSource)
        {
            return;
        }

        await _store.ClearAllAsync(cancellationToken);
        if (_exclusionStore is not null)
        {
            // Identity ordinals restart from 1 after a full clear, so stale pairs would bind to unrelated
            // new faces reusing those keys.
            await _exclusionStore.ClearAsync(cancellationToken);
        }
    }
}

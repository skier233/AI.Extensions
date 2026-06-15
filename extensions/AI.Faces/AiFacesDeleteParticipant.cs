using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Faces;

internal sealed class AiFacesDeleteParticipant(IFaceIdentityStore store) : IFaceLifecycleParticipant
{
    private const string FaceSourceKey = "ext:ai.faces";

    private readonly IFaceIdentityStore _store = store;

    public Task OnDeletingAsync(Face face, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(face.PrimarySourceKey))
        {
            return Task.CompletedTask;
        }

        return _store.DeleteByFaceKeyAsync(face.PrimarySourceKey, cancellationToken);
    }

    public Task OnFacesPurgedAsync(FacePurgeScope scope, CancellationToken cancellationToken = default)
    {
        // An "entire source" clear of this extension's (or all) face data must also drop provisional
        // identities, which have no Cove Face row and so are never reached by OnDeletingAsync. Narrower
        // purges leave the working identity graph intact (the per-face path handles promoted identities).
        var clearsThisSource = string.IsNullOrWhiteSpace(scope.SourceKey)
            || string.Equals(scope.SourceKey, FaceSourceKey, StringComparison.OrdinalIgnoreCase);
        if (scope.IsEntireSource && clearsThisSource)
        {
            return _store.ClearAllAsync(cancellationToken);
        }

        return Task.CompletedTask;
    }
}

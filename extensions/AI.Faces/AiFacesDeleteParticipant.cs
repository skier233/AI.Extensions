using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Faces;

internal sealed class AiFacesDeleteParticipant(IFaceIdentityStateStore stateStore) : IFaceLifecycleParticipant
{
    private readonly IFaceIdentityStateStore _stateStore = stateStore;

    public Task OnDeletingAsync(Face face, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(face.PrimarySourceKey))
        {
            return Task.CompletedTask;
        }

        return _stateStore.DeleteAsync(face.PrimarySourceKey, cancellationToken);
    }
}
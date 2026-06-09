namespace AI.Faces;

// Synthetic, negative performer ids used to represent an as-yet-unlinked reference identity in a
// suggestion. Negative so they never collide with real (positive) Cove performer ids. With multiple
// reference packs loaded at once a bare ordinal is no longer unique, so the id encodes both the
// pack's stable index (its position in the deterministically-ordered active pack list) and the
// identity's ordinal within that pack.
internal static class AiFaceReferenceSuggestionIds
{
    // Allows up to PackStride identities per pack. .saie packs are per-site performer sets, far below
    // this bound, and the negative-int space still covers hundreds of packs.
    private const int PackStride = 1_000_000;

    public static int FromOrdinal(int ordinal) => FromIdentity(0, ordinal);

    public static int FromIdentity(int packIndex, int ordinal)
    {
        if (packIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(packIndex));
        if (ordinal < 0 || ordinal >= PackStride)
            throw new ArgumentOutOfRangeException(nameof(ordinal));

        return checked(-((packIndex * PackStride) + ordinal + 1));
    }

    public static bool TryResolve(IReadOnlyList<SaieReferenceIdentity> identities, int suggestionId, out SaieReferenceIdentity? identity)
    {
        identity = null;
        if (suggestionId >= 0)
            return false;

        var encoded = (-suggestionId) - 1;
        var ordinal = encoded % PackStride;
        if (encoded / PackStride != 0 || ordinal < 0 || ordinal >= identities.Count)
            return false;

        identity = identities[ordinal];
        return true;
    }

    public static bool TryResolve(
        IReadOnlyList<SaieReferencePack> packs,
        int suggestionId,
        out SaieReferencePack? pack,
        out SaieReferenceIdentity? identity)
    {
        pack = null;
        identity = null;
        if (suggestionId >= 0)
            return false;

        var encoded = (-suggestionId) - 1;
        var packIndex = encoded / PackStride;
        var ordinal = encoded % PackStride;
        if (packIndex < 0 || packIndex >= packs.Count)
            return false;

        var candidate = packs[packIndex];
        if (ordinal < 0 || ordinal >= candidate.Identities.Count)
            return false;

        pack = candidate;
        identity = candidate.Identities[ordinal];
        return true;
    }
}

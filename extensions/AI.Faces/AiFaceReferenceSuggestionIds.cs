namespace AI.Faces;

internal static class AiFaceReferenceSuggestionIds
{
    public static int FromOrdinal(int ordinal)
    {
        if (ordinal < 0)
            throw new ArgumentOutOfRangeException(nameof(ordinal));

        return checked(-(ordinal + 1));
    }

    public static bool TryResolve(IReadOnlyList<SaieReferenceIdentity> identities, int suggestionId, out SaieReferenceIdentity? identity)
    {
        identity = null;
        if (suggestionId >= 0)
            return false;

        var ordinal = checked((-suggestionId) - 1);
        if (ordinal < 0 || ordinal >= identities.Count)
            return false;

        identity = identities[ordinal];
        return true;
    }
}
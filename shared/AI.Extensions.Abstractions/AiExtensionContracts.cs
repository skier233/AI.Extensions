namespace AI.Extensions.Abstractions;

public static class AiMediaKinds
{
    public const string Image = "image";
    public const string Video = "video";
    public const string Audio = "audio";
}

public static class AiLoadPolicies
{
    public const string UseLoaded = "use_loaded";
    public const string LoadIfCheap = "load_if_cheap";
    public const string LoadOrFail = "load_or_fail";

    public static IReadOnlySet<string> All { get; } = new HashSet<string>(StringComparer.Ordinal)
    {
        UseLoaded,
        LoadIfCheap,
        LoadOrFail,
    };
}

public sealed record AiCapabilityClaim(
    string ClaimId,
    string DisplayName,
    string MediaKind,
    string WantCapability,
    string WantScope,
    string OutputKey,
    IReadOnlyList<string>? PreferredModels = null,
    string? FromDetection = null,
    string? Description = null
)
{
    public string? CapabilityId { get; init; }

    public string? ModelBindingSlotId { get; init; }
}

public sealed record AiModelBindingSlot(
    string SlotId,
    string DisplayName,
    string WantCapability,
    IReadOnlyList<string>? RequiredCapabilities = null,
    IReadOnlyList<string>? RequiredScopes = null,
    IReadOnlyList<string>? RequiredCategories = null,
    IReadOnlyList<string>? DefaultModels = null,
    bool CategoryScoped = false,
    bool AllowMultiple = false,
    string? Description = null
);

public sealed record AiCapabilityFeature(
    string CapabilityId,
    string DisplayName,
    IReadOnlyList<string> ClaimIds,
    IReadOnlyList<AiModelBindingSlot>? ModelBindingSlots = null,
    string? Description = null
);

public sealed record AiCapabilityDescriptor(
    string ExtensionId,
    string DisplayName,
    IReadOnlyList<AiCapabilityClaim> Claims
)
{
    public IReadOnlyList<AiCapabilityFeature> Capabilities { get; init; } = [];
}

public sealed record AiDispatchRequest(
    AiRunContext Context,
    IReadOnlyList<AiCapabilityClaim> Claims,
    AiAnalyzeResult Result,
    IReadOnlyDictionary<string, string>? Metadata = null
);

public sealed record AiDispatchResult(
    string ExtensionId,
    int ClaimCount,
    IReadOnlyDictionary<string, int>? PreparedCounts = null,
    IReadOnlyList<string>? Notes = null
);

public interface IAiCapabilityContributor
{
    AiCapabilityDescriptor Describe();

    Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default);

    /// <summary>
    /// Dispatches a whole batch of per-entity results to this contributor. The default implementation simply
    /// calls <see cref="DispatchAsync"/> for each request, so existing contributors keep working unchanged.
    /// Contributors that persist to the database (embeddings, tags) should override this to write the whole
    /// batch in a single unit of work — one scope, bulk queries, one SaveChanges — instead of one transaction
    /// per entity, which is what made large image batches slow.
    /// </summary>
    async Task<IReadOnlyList<AiDispatchResult>> DispatchBatchAsync(IReadOnlyList<AiDispatchRequest> requests, CancellationToken ct = default)
    {
        var results = new List<AiDispatchResult>(requests.Count);
        foreach (var request in requests)
        {
            results.Add(await DispatchAsync(request, ct));
        }

        return results;
    }
}

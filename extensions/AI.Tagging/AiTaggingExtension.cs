using AI.Extensions.Abstractions;

using Cove.Plugins;
using Cove.Sdk;

using Microsoft.Extensions.DependencyInjection;

namespace AI.Tagging;

public sealed class AiTaggingExtension : CoveExtensionBase
{
    public override string Id => "cove.ai.tagging";

    public override string Name => "AI Tagging";

    public override string Version => "0.1.0";

    public override string Description => "Contributes tagging claims for image and video AI workflows.";

    public override string Author => "Cove Team";

    public override string Url => "https://github.com/yourcove/AI.Extensions";

    public override string MinCoveVersion => "0.0.10";

    public override IReadOnlyList<string> Categories =>
    [
        ExtensionCategories.Metadata,
        ExtensionCategories.Automation,
        "ai",
        "tagging",
    ];

    public override IReadOnlyDictionary<string, string> Dependencies => new Dictionary<string, string>
    {
        ["cove.ai.core"] = ">=0.1.0",
    };

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        services.AddSingleton<AiTaggingPreparationService>();
        services.AddSingleton<AiTaggingPersistenceService>();
        services.AddSingleton<IAiCapabilityContributor, AiTaggingContributor>();
    }
}

internal sealed class AiTaggingContributor(
    AiTaggingPreparationService preparationService,
    AiTaggingPersistenceService persistenceService) : IAiCapabilityContributor
{
    private readonly AiTaggingPreparationService _preparationService = preparationService;
    private readonly AiTaggingPersistenceService _persistenceService = persistenceService;

    private static readonly AiCapabilityDescriptor Descriptor = new(
        "cove.ai.tagging",
        "AI Tagging",
        [
            new AiCapabilityClaim(
                "tagging.image.asset",
                "Image Tags",
                AiMediaKinds.Image,
                "tagging",
                "asset",
                "tags",
                Description: "Generate asset-level tags for still images."),
            new AiCapabilityClaim(
                "tagging.video.frame",
                "Video Tags",
                AiMediaKinds.Video,
                "tagging",
                "frame",
                "video_tag_info",
                Description: "Generate frame-aware video tags and timeline aggregates."),
        ]);

    public AiCapabilityDescriptor Describe() => Descriptor;

    public async Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default)
    {
        var batch = _preparationService.Prepare(request);
        var notes = new List<string>(batch.Notes);
        notes.AddRange(await _persistenceService.PersistAsync(request, batch, ct));

        return new AiDispatchResult(
            Descriptor.ExtensionId,
            request.Claims.Count,
            batch.ToPreparedCounts(),
            notes);
    }
}


using AI.Extensions.Abstractions;

using Cove.Plugins;
using Cove.Sdk;

using Microsoft.Extensions.DependencyInjection;

namespace AI.Audio;

public sealed class AiAudioExtension : CoveExtensionBase
{
    public override string Id => "cove.ai.audio";

    public override string Name => "AI Audio";

    public override string Version => "0.1.0";

    public override string Description => "Contributes audio embedding and classification claims for AI workflows.";

    public override string Author => "Cove Team";

    public override string Url => "https://github.com/yourcove/AI.Extensions";

    public override string MinCoveVersion => "0.0.10";

    public override IReadOnlyList<string> Categories =>
    [
        ExtensionCategories.Metadata,
        ExtensionCategories.Automation,
        "ai",
        "audio",
    ];

    public override IReadOnlyDictionary<string, string> Dependencies => new Dictionary<string, string>
    {
        ["cove.ai.core"] = ">=0.1.0",
    };

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        services.AddSingleton<AiAudioPreparationService>();
        services.AddSingleton<AiAudioPersistenceService>();
        services.AddSingleton<IAiCapabilityContributor, AiAudioContributor>();
    }
}

internal sealed class AiAudioContributor(
    AiAudioPreparationService preparationService,
    AiAudioPersistenceService persistenceService) : IAiCapabilityContributor
{
    private readonly AiAudioPreparationService _preparationService = preparationService;
    private readonly AiAudioPersistenceService _persistenceService = persistenceService;

    private static readonly AiCapabilityDescriptor Descriptor = new(
        "cove.ai.audio",
        "AI Audio",
        [
            new AiCapabilityClaim(
                "audio.asset.embedding",
                "Audio Embeddings",
                AiMediaKinds.Audio,
                "embedding",
                "asset",
                "embeddings",
                Description: "Extract audio embeddings for speaker and similarity workflows."),
            new AiCapabilityClaim(
                "audio.asset.classification",
                "Audio Classification",
                AiMediaKinds.Audio,
                "classification",
                "asset",
                "categories",
                Description: "Classify audio content for filtering and downstream routing."),
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


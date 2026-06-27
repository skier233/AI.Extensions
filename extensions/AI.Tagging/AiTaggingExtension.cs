using AI.Extensions.Abstractions;

using Cove.Core.Auth;
using Cove.Core.Interfaces;
using Cove.Plugins;
using Cove.Sdk;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Tagging;

public sealed class AiTaggingExtension : FullExtensionBase, IPermissionContributor
{
    public const string WriteSettingsPermission = "cove.community.ai.tagging.settings.write";

    public override string Id => "cove.community.ai.tagging";

    public override string Name => "AI Tagging";

    public override string Version => "0.3.0";

    public override string Description => "Contributes tagging claims for image and video AI workflows.";

    public override string Author => "skier233";

    public override string Url => "https://github.com/skier233/AI.Extensions";

    public override string MinCoveVersion => "0.6.0";

    public override IReadOnlyList<string> Categories =>
    [
        ExtensionCategories.Metadata,
        ExtensionCategories.Automation,
        "ai",
        "tagging",
    ];

    public override IReadOnlyDictionary<string, string> Dependencies => new Dictionary<string, string>
    {
        ["cove.community.ai.core"] = ">=0.3.0",
    };

    public override UIManifest GetUIManifest()
        => ManifestBuilder()
            .AddSettingsTab(
                "extensions/ai/tagging",
                "AI Tagging",
                order: 110,
                icon: "tags",
                parentTabKey: "extensions/ai",
                description: "AI tagging extension settings.",
                searchKeywords: ["ai tagging", "tags", "tag overrides", "nsfw ai server"],
                aliases: ["extensions-ai-tagging"])
            .AddSettingsSection("extensions/ai/tagging", "AI Tagging", "AiTaggingSettingsPanel", order: 50)
            .Build();

    public IEnumerable<PermissionDefinition> ContributePermissions()
    {
        var source = $"extension:{Id}";
        return
        [
            new(WriteSettingsPermission, "AI Tagging", "Change AI-generated tag name overrides.", Dangerous: true, Source: source, GrantToAdminsByDefault: true),
        ];
    }

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        services.AddSingleton<AiTaggingPreparationService>();
        services.AddSingleton<AiTaggingPersistenceService>();
        services.AddSingleton<IAiCapabilityContributor, AiTaggingContributor>();
    }

    public override Task InitializeAsync(IServiceProvider services, CancellationToken ct = default)
    {
        PublishContributions<IAiCapabilityContributor>(services);
        return Task.CompletedTask;
    }

    public override void MapEndpoints(IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("/api/ext/ai-tagging").WithTags("AI.Tagging");

        group.MapGet("/settings", async (CancellationToken ct) => Results.Ok(await AiTaggingSettingsStore.LoadAsync(Store, ct)));

        group.MapPut("/settings", async (AiTaggingSettings settings, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, WriteSettingsPermission) is { } denied)
                return denied;

            try
            {
                var normalized = settings.Normalize();
                await AiTaggingSettingsStore.SaveAsync(Store, normalized, ct);
                return Results.Ok(normalized);
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
            }
        });
    }

    private static IResult? RequirePermission(ICurrentPrincipalAccessor principalAccessor, string permission)
    {
        var principal = principalAccessor.Current;
        if (principal?.Has(permission) == true)
            return null;

        return Results.Problem("Forbidden", statusCode: StatusCodes.Status403Forbidden);
    }
}

internal sealed class AiTaggingContributor(
    AiTaggingPreparationService preparationService,
    AiTaggingPersistenceService persistenceService) : IAiCapabilityContributor
{
    private readonly AiTaggingPreparationService _preparationService = preparationService;
    private readonly AiTaggingPersistenceService _persistenceService = persistenceService;

    private static readonly AiCapabilityDescriptor Descriptor = new(
        "cove.community.ai.tagging",
        "AI Tagging",
        [
            new AiCapabilityClaim(
                "tagging.image.asset",
                "Image Tags",
                AiMediaKinds.Image,
                "tagging",
                "asset",
                "tags",
                Description: "Generate asset-level tags for still images.")
            {
                CapabilityId = "tagging",
                ModelBindingSlotId = "category",
            },
            new AiCapabilityClaim(
                "tagging.video.frame",
                "Video Tags",
                AiMediaKinds.Video,
                "tagging",
                "frame",
                "video_tag_info",
                Description: "Generate frame-aware video tags and timeline aggregates.")
            {
                CapabilityId = "tagging",
                ModelBindingSlotId = "category",
            },
        ])
    {
        Capabilities =
        [
            new AiCapabilityFeature(
                "tagging",
                "Content Tagging",
                ["tagging.image.asset", "tagging.video.frame"],
                [
                    new AiModelBindingSlot(
                        "category",
                        "Tagging category model",
                        "tagging",
                        RequiredCapabilities: ["tagging"],
                        RequiredScopes: ["asset", "frame"],
                        CategoryScoped: true,
                        Description: "One binding per discovered model category; new server categories do not require extension code changes."),
                ],
                "Generate tags from every discovered tagging category."),
        ],
    };

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


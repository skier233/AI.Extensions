using AI.Extensions.Abstractions;

using System.IO;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Routing;

using Cove.Core.Auth;
using Cove.Core.Interfaces;
using Cove.Data;
using Cove.Data.Services;
using Cove.Plugins;
using Cove.Sdk;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.EntityFrameworkCore;

namespace AI.Faces;

public sealed class AiFacesExtension : FullExtensionBase, IPermissionContributor
{
    public const string WriteSettingsPermission = "cove.ai.faces.settings.write";
    public const string UploadReferencePermission = "cove.ai.faces.reference.upload";
    public const string DeleteReferencePermission = "cove.ai.faces.reference.delete";
    public const string ApplyReferencePermission = "cove.ai.faces.reference.apply";

    public override string Id => "cove.ai.faces";

    public override string Name => "AI Faces";

    public override string Version => "0.1.0";

    public override string Description => "Contributes face-region and face-embedding claims for AI workflows.";

    public override string Author => "Cove Team";

    public override string Url => "https://github.com/yourcove/AI.Extensions";

    public override string MinCoveVersion => "0.0.10";

    public override IReadOnlyList<string> Categories =>
    [
        ExtensionCategories.Metadata,
        ExtensionCategories.Automation,
        "ai",
        "faces",
    ];

    public override IReadOnlyDictionary<string, string> Dependencies => new Dictionary<string, string>
    {
        ["cove.ai.core"] = ">=0.1.0",
    };

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        var referenceRoot = Path.Combine(context.DataDirectory, Id, "reference");

        services.AddSingleton<StoreBackedFaceIdentityStateStore>();
        services.AddSingleton<IFaceIdentityStateStore>(static services => services.GetRequiredService<StoreBackedFaceIdentityStateStore>());
        services.AddSingleton<StoreBackedAiFacesSettingsStore>();
        services.AddSingleton<IAiFacesSettingsStore>(static services => services.GetRequiredService<StoreBackedAiFacesSettingsStore>());
        services.AddSingleton<SaieArchiveReader>();
        services.AddSingleton(services => new AiFaceReferencePackStore(referenceRoot, services.GetRequiredService<SaieArchiveReader>()));
        services.AddSingleton<AiFaceReferenceSuggestionDecisionStore>();
        services.AddSingleton<IFaceLifecycleParticipant, AiFacesDeleteParticipant>();
        services.AddSingleton<AiAssetFaceClusterer>();
        services.AddSingleton<AiFaceIdentityReconciler>();
        services.AddSingleton<AiFaceReferenceBackfillService>();
        services.AddSingleton<AiFacePreparationService>();
        services.AddSingleton<AiFacesPersistenceService>();
        services.AddSingleton<IAiCapabilityContributor, AiFacesContributor>();
        services.AddScoped<AiFaceReferencePerformerResolver>();
        services.AddScoped<IFaceSuggester, AiFaceSuggester>();
    }

    public override Task InitializeAsync(IServiceProvider services, CancellationToken ct = default)
    {
        services.GetRequiredService<StoreBackedFaceIdentityStateStore>().Attach(Store);
        services.GetRequiredService<StoreBackedAiFacesSettingsStore>().Attach(Store);
        AiFacesSettingsRuntime.Attach(services.GetRequiredService<IAiFacesSettingsStore>());
        services.GetRequiredService<AiFaceReferencePackStore>().Attach(Store);
        services.GetRequiredService<AiFaceReferenceSuggestionDecisionStore>().Attach(Store);
        return Task.CompletedTask;
    }

    public override UIManifest GetUIManifest()
        => ManifestBuilder()
            .AddSettingsSection("ai-data", "AI Faces", "AiFacesSettingsPanel", order: 60)
            .Build();

    public IEnumerable<PermissionDefinition> ContributePermissions()
    {
        var source = $"extension:{Id}";
        return
        [
            new(WriteSettingsPermission, "AI Faces", "Change AI Faces settings.", Dangerous: true, Source: source),
            new(UploadReferencePermission, "AI Faces", "Upload and import AI.Faces .saie reference packs.", Dangerous: true, Source: source),
            new(DeleteReferencePermission, "AI Faces", "Remove the active AI.Faces reference pack.", Dangerous: true, Source: source),
            new(ApplyReferencePermission, "AI Faces", "Accept or reject AI.Faces reference identity suggestions.", Dangerous: true, Implies: [Cove.Core.Auth.Permissions.FacesWrite, Cove.Core.Auth.Permissions.PerformersWrite], Source: source),
        ];
    }

    public override void MapEndpoints(IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("/api/ext/ai-faces").WithTags("AI.Faces");

        group.MapGet("/settings", async (IAiFacesSettingsStore settingsStore, CancellationToken ct) =>
            Results.Ok(await settingsStore.LoadAsync(ct)));

        group.MapPut("/settings", async (AiFacesSettings settings, IAiFacesSettingsStore settingsStore, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, WriteSettingsPermission) is { } denied)
                return denied;

            try
            {
                var normalized = settings.Normalize();
                await settingsStore.SaveAsync(normalized, ct);
                return Results.Ok(normalized);
            }
            catch (Exception ex)
            {
                return Results.Problem(
                    title: "AI.Faces settings update failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError);
            }
        });

        group.MapGet("/reference/status", async (AiFaceReferencePackStore packStore, CancellationToken ct) =>
        {
            var status = await packStore.GetStatusAsync(ct);
            return Results.Ok(status);
        });

        group.MapDelete("/reference", async (AiFaceReferencePackStore packStore, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, DeleteReferencePermission) is { } denied)
                return denied;

            await packStore.ClearAsync(ct);
            return Results.NoContent();
        });

        group.MapPost("/reference/import", async (HttpRequest request, IJobService jobService, AiFaceReferencePackStore packStore, AiFaceReferenceBackfillService backfillService, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
            {
                if (RequirePermission(principalAccessor, UploadReferencePermission) is { } denied)
                    return denied;

                if (!request.HasFormContentType)
                    return Results.BadRequest(new { error = "Upload must use multipart/form-data." });

                var form = await request.ReadFormAsync(ct);
                var file = form.Files.Count > 0 ? form.Files[0] : null;
                if (file is null || file.Length <= 0)
                    return Results.BadRequest(new { error = "A .saie file is required." });

                if (!file.FileName.EndsWith(".saie", StringComparison.OrdinalIgnoreCase))
                    return Results.BadRequest(new { error = "Only .saie reference packs are supported." });

                await using var uploadStream = file.OpenReadStream();
                var stagedPath = await packStore.StageUploadAsync(uploadStream, file.FileName, ct);
                var originalFileName = Path.GetFileName(file.FileName);

                var jobId = jobService.Enqueue(
                    "ai-faces-reference-import",
                    $"Importing AI.Faces reference pack {originalFileName}",
                    async (progress, jobCt) =>
                    {
                        await packStore.ImportStagedAsync(stagedPath, originalFileName, progress, jobCt);
                        var pack = await packStore.GetActivePackAsync(jobCt);
                        if (pack is null)
                        {
                            return;
                        }

                        progress.Report(0.98, "Reconciling existing face identities...");
                        await backfillService.BackfillAsync(pack, jobCt);
                    },
                    exclusive: false);

                return Results.Accepted($"/api/jobs/{jobId}", new { jobId });
            })
            .WithMetadata(
                new RequestSizeLimitAttribute(512L * 1024 * 1024),
                new RequestFormLimitsAttribute { MultipartBodyLengthLimit = 512L * 1024 * 1024 });

        group.MapPost("/reference/faces/{faceId:int}/reject", async (
            int faceId,
            AiFaceReferenceSuggestionDecisionRequest request,
            CoveContext db,
            AiFaceReferencePackStore packStore,
            AiFaceReferenceSuggestionDecisionStore decisionStore,
            ICurrentPrincipalAccessor principalAccessor,
            CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, ApplyReferencePermission) is { } denied)
                return denied;

            var faceExists = await db.Faces.AsNoTracking().AnyAsync(face => face.Id == faceId, ct);
            if (!faceExists)
                return Results.NotFound();

            var pack = await packStore.GetActivePackAsync(ct);
            if (pack is null || !AiFaceReferenceSuggestionIds.TryResolve(pack.Identities, request.ReferenceSuggestionId, out var identity) || identity is null)
                return Results.BadRequest(new { error = "referenceSuggestionId did not resolve to an imported reference identity." });

            await decisionStore.RejectAsync(faceId, identity.ExternalId, ct);
            return Results.NoContent();
        });

        group.MapPost("/reference/faces/{faceId:int}/import-performer", async (
            int faceId,
            AiFaceReferenceSuggestionDecisionRequest request,
            CoveContext db,
            AiFaceReferencePackStore packStore,
            AiFaceReferencePerformerResolver performerResolver,
            AiFaceReferenceSuggestionDecisionStore decisionStore,
            FacePerformerPropagationService facePerformerPropagationService,
            ICurrentPrincipalAccessor principalAccessor,
            CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, ApplyReferencePermission) is { } denied)
                return denied;

            var face = await db.Faces.FirstOrDefaultAsync(item => item.Id == faceId, ct);
            if (face is null)
                return Results.NotFound();

            var pack = await packStore.GetActivePackAsync(ct);
            var status = await packStore.GetStatusAsync(ct);
            if (pack is null || status is null)
                return Results.BadRequest(new { error = "Import a reference .saie pack before linking reference identities." });

            if (!AiFaceReferenceSuggestionIds.TryResolve(pack.Identities, request.ReferenceSuggestionId, out var identity) || identity is null)
                return Results.BadRequest(new { error = "referenceSuggestionId did not resolve to an imported reference identity." });

            var performer = await performerResolver.FindOrCreateAsync(identity, status.SourceEndpoint, ct);
            await decisionStore.ClearAsync(faceId, identity.ExternalId, ct);
            await facePerformerPropagationService.ApplyLinkChangeAsync(faceId, face.PerformerId, performer.Id, ct);
            face.PerformerId = performer.Id;
            await db.SaveChangesAsync(ct);

            return Results.Ok(new { performerId = performer.Id, performerName = performer.Name });
        });
    }

    private static IResult? RequirePermission(ICurrentPrincipalAccessor principalAccessor, string permission)
    {
        var principal = principalAccessor.Current;
        if (principal?.Has(permission) == true)
            return null;

        if (principal is null || principal.Kind == PrincipalKind.Anonymous)
            return Results.Unauthorized();

        return Results.Json(
            new { code = "FORBIDDEN", message = $"Permission '{permission}' is required." },
            statusCode: StatusCodes.Status403Forbidden);
    }
}

internal sealed class AiFacesContributor(
    AiFacePreparationService preparationService,
    AiFacesPersistenceService persistenceService) : IAiCapabilityContributor
{
    private const string FaceDetectionModel = "face_detector_torchexport";
    private const string FaceEmbeddingModel = "face_embedding_torchexport";

    private readonly AiFacePreparationService _preparationService = preparationService;
    private readonly AiFacesPersistenceService _persistenceService = persistenceService;

    private static readonly AiCapabilityDescriptor Descriptor = new(
        "cove.ai.faces",
        "AI Faces",
        [
            new AiCapabilityClaim(
                "faces.image.detection",
                "Image Face Detection",
                AiMediaKinds.Image,
                "detection",
                "asset",
                "regions",
                PreferredModels: [FaceDetectionModel],
                Description: "Detect faces in still images so face embeddings can be attached to concrete regions."),
            new AiCapabilityClaim(
                "faces.image.embedding",
                "Image Face Identity Embeddings",
                AiMediaKinds.Image,
                "embedding",
                "region",
                "regions",
                PreferredModels: [FaceEmbeddingModel],
                FromDetection: FaceDetectionModel,
                Description: "Extract face-region embeddings from still images."),
            new AiCapabilityClaim(
                "faces.video.detection",
                "Video Face Detection",
                AiMediaKinds.Video,
                "detection",
                "frame",
                "frames",
                PreferredModels: [FaceDetectionModel],
                Description: "Detect faces across sampled video frames before identity matching."),
            new AiCapabilityClaim(
                "faces.video.embedding",
                "Video Face Identity Embeddings",
                AiMediaKinds.Video,
                "embedding",
                "region",
                "regions",
                PreferredModels: [FaceEmbeddingModel],
                FromDetection: FaceDetectionModel,
                Description: "Extract face-region embeddings across analyzed video frames."),
        ]);

    public AiCapabilityDescriptor Describe() => Descriptor;

    public async Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default)
    {
        var batch = await _preparationService.PrepareAsync(request, ct);
        var notes = new List<string>(batch.Notes);
        notes.AddRange(await _persistenceService.PersistAsync(request, batch, ct));

        return new AiDispatchResult(
            Descriptor.ExtensionId,
            request.Claims.Count,
            batch.ToPreparedCounts(),
            notes);
    }
}


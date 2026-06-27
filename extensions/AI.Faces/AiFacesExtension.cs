using AI.Extensions.Abstractions;

using System.IO;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Routing;

using Cove.Core.Auth;
using Cove.Core.Interfaces;
using Cove.Plugins;
using Cove.Sdk;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Faces;

public sealed class AiFacesExtension : FullExtensionBase, IPermissionContributor
{
    public const string WriteSettingsPermission = "cove.community.ai.faces.settings.write";
    public const string UploadReferencePermission = "cove.community.ai.faces.reference.upload";
    public const string DeleteReferencePermission = "cove.community.ai.faces.reference.delete";
    public const string ApplyReferencePermission = "cove.community.ai.faces.reference.apply";

    public override string Id => "cove.community.ai.faces";

    public override string Name => "AI Faces";

    public override string Version => "0.3.0";

    public override string Description => "Contributes face-region and face-embedding claims for AI workflows.";

    public override string Author => "skier233";

    public override string Url => "https://github.com/skier233/AI.Extensions";

    public override string MinCoveVersion => "0.6.0";

    public override IReadOnlyList<string> Categories =>
    [
        ExtensionCategories.Metadata,
        ExtensionCategories.Automation,
        "ai",
        "faces",
    ];

    public override IReadOnlyDictionary<string, string> Dependencies => new Dictionary<string, string>
    {
        ["cove.community.ai.core"] = ">=0.3.0",
    };

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        // Persist reference packs outside the extension's own (id-named) install directory. That
        // directory is replaced wholesale on an extension update, which previously deleted imported
        // .saie packs; the ".ai-faces-data" sibling under the shared extensions data root survives
        // updates. The legacy location is passed so existing single-pack installs migrate forward.
        var legacyReferenceRoot = Path.Combine(context.DataDirectory, Id, "reference");
        var referenceRoot = Path.Combine(context.DataDirectory, ".ai-faces-data", "reference");

        // The legacy blob store is retained only so DbFaceIdentityStore can perform the one-time import of
        // the old `face-identity-snapshot` into the relational identity tables.
        services.AddSingleton<StoreBackedFaceIdentityStateStore>();
        services.AddSingleton<IFaceIdentityStateStore>(static services => services.GetRequiredService<StoreBackedFaceIdentityStateStore>());
        services.AddSingleton<IFaceIdentityStore, DbFaceIdentityStore>();
        services.AddSingleton<StoreBackedAiFacesSettingsStore>();
        services.AddSingleton<IAiFacesSettingsStore>(static services => services.GetRequiredService<StoreBackedAiFacesSettingsStore>());
        services.AddSingleton<SaieArchiveReader>();
        services.AddSingleton(services => new AiFaceReferencePackStore(referenceRoot, services.GetRequiredService<SaieArchiveReader>(), legacyReferenceRoot));
        services.AddSingleton<AiFaceReferenceSuggestionDecisionStore>();
        services.AddSingleton<AiFacePresenceSuppressionStore>();
        services.AddSingleton<AiFaceNotPresentService>();
        services.AddSingleton<IFaceLifecycleParticipant, AiFacesDeleteParticipant>();
        services.AddSingleton<AiAssetFaceClusterer>();
        services.AddSingleton<AiFaceIdentityReconciler>();
        services.AddSingleton<AiFaceReferenceBackfillService>();
        services.AddSingleton<AiFacePreparationService>();
        services.AddSingleton<AiFacesPersistenceService>();
        services.AddSingleton<IAiCapabilityContributor, AiFacesContributor>();
        services.AddScoped<AiFaceReferencePerformerResolver>();
        // Register the real scoped implementations as concrete types, then expose them across the
        // extension-isolation boundary as singleton bridges that the host consumes via the service
        // exchange (see ExtensionFaceContributions and InitializeAsync's PublishContributions calls).
        services.AddScoped<AiFaceReferenceSuggestionDecisionHandler>();
        services.AddSingleton<IFaceSuggestionDecisionHandler, ExtensionFaceSuggestionDecisionHandler>();
        services.AddScoped<AiFaceSuggester>();
        services.AddSingleton<IFaceSuggester, ExtensionFaceSuggester>();
    }

    public override Task InitializeAsync(IServiceProvider services, CancellationToken ct = default)
    {
        PublishContributions<IAiCapabilityContributor>(services);
        // These contracts are consumed by Cove host services (FacesController, AiDataPurgeService),
        // which since the extensions-runtime redesign can only see sibling/extension services through
        // the cross-extension exchange. Without publishing them the host falls back to the empty
        // suggester and no face ever lists a suggestion.
        PublishContributions<IFaceSuggester>(services);
        PublishContributions<IFaceSuggestionDecisionHandler>(services);
        PublishContributions<IFaceLifecycleParticipant>(services);
        services.GetRequiredService<StoreBackedFaceIdentityStateStore>().Attach(Store);
        services.GetRequiredService<StoreBackedAiFacesSettingsStore>().Attach(Store);
        AiFacesSettingsRuntime.Attach(services.GetRequiredService<IAiFacesSettingsStore>());
        services.GetRequiredService<AiFaceReferencePackStore>().Attach(Store);
        services.GetRequiredService<AiFaceReferenceSuggestionDecisionStore>().Attach(Store);
        services.GetRequiredService<AiFacePresenceSuppressionStore>().Attach(Store);
        return Task.CompletedTask;
    }

    public override UIManifest GetUIManifest()
        => ManifestBuilder()
            .AddSettingsTab(
                "extensions/ai/faces",
                "AI Faces",
                order: 120,
                icon: "database",
                parentTabKey: "extensions/ai",
                description: "Face recognition extension settings.",
                searchKeywords: ["ai faces", "faces", "reference pack", "clustering", "identity"],
                aliases: ["extensions-ai-faces"])
            .AddSettingsSection("extensions/ai/faces", "AI Faces", "AiFacesSettingsPanel", order: 60)
            .Build();

    public IEnumerable<PermissionDefinition> ContributePermissions()
    {
        var source = $"extension:{Id}";
        return
        [
            new(WriteSettingsPermission, "AI Faces", "Change AI Faces settings.", Dangerous: true, Source: source, GrantToAdminsByDefault: true),
            new(UploadReferencePermission, "AI Faces", "Upload and import AI.Faces .saie reference packs.", Dangerous: true, Source: source, GrantToAdminsByDefault: true),
            new(DeleteReferencePermission, "AI Faces", "Remove the active AI.Faces reference pack.", Dangerous: true, Source: source, GrantToAdminsByDefault: true),
            new(ApplyReferencePermission, "AI Faces", "Accept or reject AI.Faces reference identity suggestions.", Dangerous: true, Implies: [Cove.Core.Auth.Permissions.FacesWrite, Cove.Core.Auth.Permissions.PerformersWrite], Source: source, GrantToAdminsByDefault: true),
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

        // Back-compat: returns the first active pack as a single status object, matching the original
        // single-pack contract the settings panel consumes.
        group.MapGet("/reference/status", async (AiFaceReferencePackStore packStore, CancellationToken ct) =>
            Results.Ok(await packStore.GetStatusAsync(ct)));

        // Full multi-pack listing (one entry per imported site pack).
        group.MapGet("/reference/packs", async (AiFaceReferencePackStore packStore, CancellationToken ct) =>
            Results.Ok(await packStore.GetStatusesAsync(ct)));

        group.MapDelete("/reference", async (string? packId, AiFaceReferencePackStore packStore, IFaceTopSuggestionMaintenance suggestionMaintenance, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, DeleteReferencePermission) is { } denied)
                return denied;

            // packId omitted -> clear every pack; otherwise remove just that site's pack.
            await packStore.ClearAsync(packId, ct);
            // Removing a pack changes every unlinked face's reference matches, so invalidate the
            // materialized top-suggestion projection; the host recomputes it off the request path.
            await suggestionMaintenance.InvalidateAllUnlinkedAsync(ct);
            return Results.NoContent();
        });

        group.MapPost("/reference/import", async (HttpRequest request, IJobService jobService, AiFaceReferencePackStore packStore, AiFaceReferenceBackfillService backfillService, IServiceScopeFactory scopeFactory, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
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
                        var status = await packStore.ImportStagedAsync(stagedPath, originalFileName, progress, jobCt);
                        var packs = await packStore.GetActivePacksAsync(jobCt);
                        var pack = packs.FirstOrDefault(candidate => string.Equals(candidate.Manifest.PackId, status.PackId, StringComparison.OrdinalIgnoreCase));
                        if (pack is null)
                        {
                            return;
                        }

                        progress.Report(0.98, "Reconciling existing face identities...");
                        await backfillService.BackfillAsync(pack, jobCt);

                        // A new/updated pack changes every unlinked face's reference matches. Invalidate
                        // the materialized top-suggestion projection so the host recomputes it off the
                        // request path. Resolve from a fresh scope since the job outlives the request.
                        using var maintenanceScope = scopeFactory.CreateScope();
                        var suggestionMaintenance = maintenanceScope.ServiceProvider.GetService<IFaceTopSuggestionMaintenance>();
                        if (suggestionMaintenance is not null)
                            await suggestionMaintenance.InvalidateAllUnlinkedAsync(jobCt);
                    },
                    exclusive: false);

                return Results.Accepted($"/api/jobs/{jobId}", new { jobId });
            })
            .WithMetadata(
                new RequestSizeLimitAttribute(512L * 1024 * 1024),
                new RequestFormLimitsAttribute { MultipartBodyLengthLimit = 512L * 1024 * 1024 });

        // Mark a face as not actually present on a video/image. Splits the wrong-person occurrences off
        // the face (re-homing them to a matching or new face) and records a durable suppression.
        group.MapPost("/faces/{faceId:int}/not-present", async (
            int faceId,
            AiFaceNotPresentRequest body,
            AiFaceNotPresentService notPresentService,
            ICurrentPrincipalAccessor principalAccessor,
            CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, Cove.Core.Auth.Permissions.FacesWrite) is { } denied)
                return denied;

            if (body is null || string.IsNullOrWhiteSpace(body.HostType) || body.HostId <= 0)
                return Results.BadRequest(new { error = "hostType and a positive hostId are required." });

            var result = await notPresentService.MarkNotPresentAsync(faceId, body.HostType, body.HostId, ct);
            if (!result.FaceFound)
                return Results.NotFound(new { error = "Face was not found." });
            if (!result.HostHadFace)
                return Results.BadRequest(new { error = "That face is not present on the specified host." });

            return Results.Ok(result);
        });
    }

    // Extension-owned identity graph schema. Maps the persistence entities into the host CoveContext
    // (queried via the injected DbContext) and creates the backing tables via a raw migration. The
    // provisional/promoted identity graph used to live in a single serialized `face-identity-snapshot`
    // blob, which had to be fully loaded, globally re-merged, and re-saved on every asset — O(N^2+) per
    // asset and unbounded in N. These tables move it to relational, pgvector-backed storage so reconcile
    // can load only similarity candidates and persist deltas. See DbFaceIdentityStore.
    public override void ConfigureModel(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<ExtAiFacesIdentityEntity>(entity =>
        {
            entity.ToTable("ext_ai_faces_identity");
            entity.HasKey(item => item.Id);
            entity.Property(item => item.Id).HasColumnName("id");
            entity.Property(item => item.FaceKey).HasColumnName("face_key");
            entity.Property(item => item.Ordinal).HasColumnName("ordinal");
            entity.Property(item => item.Label).HasColumnName("label");
            entity.Property(item => item.LifecycleStatus).HasColumnName("lifecycle_status");
            entity.Property(item => item.PromotionReason).HasColumnName("promotion_reason");
            entity.Property(item => item.ReferenceExternalId).HasColumnName("reference_external_id");
            entity.Property(item => item.ReferenceDisplayName).HasColumnName("reference_display_name");
            entity.Property(item => item.ReferencePackId).HasColumnName("reference_pack_id");
            entity.Property(item => item.ReferenceSuggestionId).HasColumnName("reference_suggestion_id");
            entity.Property(item => item.QualityScore).HasColumnName("quality_score");
            entity.Property(item => item.CoverAssetId).HasColumnName("cover_asset_id");
            entity.Property(item => item.CoverX1).HasColumnName("cover_x1");
            entity.Property(item => item.CoverY1).HasColumnName("cover_y1");
            entity.Property(item => item.CoverX2).HasColumnName("cover_x2");
            entity.Property(item => item.CoverY2).HasColumnName("cover_y2");
            entity.Property(item => item.CoverQualityScore).HasColumnName("cover_quality_score");
            entity.Property(item => item.ObservationCount).HasColumnName("observation_count");
            entity.Property(item => item.AssetIdsJson).HasColumnName("asset_ids_json");
            entity.Property(item => item.CreatedAt).HasColumnName("created_at");
            entity.Property(item => item.UpdatedAt).HasColumnName("updated_at");
            entity.HasIndex(item => item.FaceKey).IsUnique();
            entity.HasIndex(item => item.ReferenceExternalId);
            entity.HasMany(item => item.Anchors)
                .WithOne()
                .HasForeignKey(anchor => anchor.IdentityId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        modelBuilder.Entity<ExtAiFacesIdentityAnchorEntity>(entity =>
        {
            entity.ToTable("ext_ai_faces_identity_anchor");
            entity.HasKey(item => item.Id);
            entity.Property(item => item.Id).HasColumnName("id");
            entity.Property(item => item.IdentityId).HasColumnName("identity_id");
            entity.Property(item => item.ModelKey).HasColumnName("model_key");
            entity.Property(item => item.QualityScore).HasColumnName("quality_score");
            entity.Property(item => item.Vector).HasColumnName("vector");
            entity.HasIndex(item => item.IdentityId);
        });
    }

    protected override void DefineMigrations()
    {
        // Raw SQL owns the exact column types (notably the pgvector `vector` column, which EF maps but
        // does not create). The anchor table is small (bounded by distinct people, not detections) and
        // candidate lookups are scoped to it, so a sequential cosine scan is fast; an ANN index can be
        // added later once the face embedder's dimension is pinned.
        Migration("001_create_identity_graph", """
            CREATE TABLE IF NOT EXISTS ext_ai_faces_identity (
                id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                face_key text NOT NULL,
                ordinal integer NOT NULL,
                label text NULL,
                lifecycle_status text NOT NULL,
                promotion_reason text NULL,
                reference_external_id text NULL,
                reference_display_name text NULL,
                reference_pack_id text NULL,
                reference_suggestion_id integer NULL,
                quality_score double precision NOT NULL,
                cover_asset_id text NULL,
                cover_x1 double precision NULL,
                cover_y1 double precision NULL,
                cover_x2 double precision NULL,
                cover_y2 double precision NULL,
                cover_quality_score double precision NOT NULL,
                observation_count integer NOT NULL,
                asset_ids_json text NOT NULL,
                created_at timestamp with time zone NOT NULL,
                updated_at timestamp with time zone NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ix_ext_ai_faces_identity_face_key ON ext_ai_faces_identity (face_key);
            CREATE INDEX IF NOT EXISTS ix_ext_ai_faces_identity_reference_external_id ON ext_ai_faces_identity (reference_external_id);

            CREATE TABLE IF NOT EXISTS ext_ai_faces_identity_anchor (
                id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                identity_id integer NOT NULL REFERENCES ext_ai_faces_identity (id) ON DELETE CASCADE,
                model_key text NOT NULL,
                quality_score double precision NOT NULL,
                vector vector NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_ext_ai_faces_identity_anchor_identity_id ON ext_ai_faces_identity_anchor (identity_id);
            """);
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
        "cove.community.ai.faces",
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
                Description: "Detect faces in still images so face embeddings can be attached to concrete regions.")
            {
                CapabilityId = "faces",
                ModelBindingSlotId = "detector",
            },
            new AiCapabilityClaim(
                "faces.image.embedding",
                "Image Face Identity Embeddings",
                AiMediaKinds.Image,
                "embedding",
                "region",
                "regions",
                PreferredModels: [FaceEmbeddingModel],
                FromDetection: FaceDetectionModel,
                Description: "Extract face-region embeddings from still images.")
            {
                CapabilityId = "faces",
                ModelBindingSlotId = "embedder",
            },
            new AiCapabilityClaim(
                "faces.video.detection",
                "Video Face Detection",
                AiMediaKinds.Video,
                "detection",
                "frame",
                "frames",
                PreferredModels: [FaceDetectionModel],
                Description: "Detect faces across sampled video frames before identity matching.")
            {
                CapabilityId = "faces",
                ModelBindingSlotId = "detector",
            },
            new AiCapabilityClaim(
                "faces.video.embedding",
                "Video Face Identity Embeddings",
                AiMediaKinds.Video,
                "embedding",
                "region",
                "regions",
                PreferredModels: [FaceEmbeddingModel],
                FromDetection: FaceDetectionModel,
                Description: "Extract face-region embeddings across analyzed video frames.")
            {
                CapabilityId = "faces",
                ModelBindingSlotId = "embedder",
            },
        ])
    {
        Capabilities =
        [
            new AiCapabilityFeature(
                "faces",
                "Facial Recognition",
                ["faces.image.detection", "faces.image.embedding", "faces.video.detection", "faces.video.embedding"],
                [
                    new AiModelBindingSlot(
                        "detector",
                        "Face detector",
                        "detection",
                        RequiredCapabilities: ["detection"],
                        RequiredScopes: ["asset", "frame"],
                        RequiredCategories: ["face_detections"],
                        DefaultModels: [FaceDetectionModel]),
                    new AiModelBindingSlot(
                        "embedder",
                        "Face embedder",
                        "embedding",
                        RequiredCapabilities: ["embedding"],
                        RequiredScopes: ["region"],
                        RequiredCategories: ["face_embeddings"],
                        DefaultModels: [FaceEmbeddingModel]),
                ],
                "Detect and identify faces as one atomic workflow."),
        ],
    };

    public AiCapabilityDescriptor Describe() => Descriptor;

    public async Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default)
    {
        var outcome = await _preparationService.PrepareWithReportAsync(request, ct);
        var batch = outcome.Batch;
        var notes = new List<string>(batch.Notes);
        notes.AddRange(await _persistenceService.PersistAsync(request, batch, outcome.MergedFaceKeyMap, ct));

        return new AiDispatchResult(
            Descriptor.ExtensionId,
            request.Claims.Count,
            batch.ToPreparedCounts(),
            notes);
    }
}


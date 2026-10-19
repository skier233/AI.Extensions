using Cove.Core.Auth;
using Cove.Core.Interfaces;
using Cove.Plugins;
using Cove.Sdk;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Review;

public sealed class AiReviewExtension : FullExtensionBase, IPermissionContributor
{
    public const string ReviewPermission = "cove.community.ai.review.review";

    private string _dataRoot = string.Empty;

    // Id, Name, Version and other metadata are sourced from extension.json by FullExtensionBase.

    public override UIManifest GetUIManifest()
        => ManifestBuilder()
            .AddPage(
                "ai-review",
                "AI Review",
                "AiReviewPage",
                icon: "list-checks",
                showInNav: true,
                navOrder: 86)
            .AddTab(
                "video",
                "ai-review",
                "AI Review",
                "AiReviewVideoTab",
                order: 95,
                countEndpoint: "/api/ext/ai-review/videos/{entityId}/count",
                icon: "list-checks")
            .AddSettingsTab(
                "extensions/ai/review",
                "AI Review",
                order: 130,
                icon: "list-checks",
                parentTabKey: "extensions/ai",
                description: "Review AI tag segments, delete false positives and export frames for training.",
                searchKeywords: ["ai review", "false positives", "segments", "training export"],
                aliases: ["extensions-ai-review"])
            .AddSettingsSection("extensions/ai/review", "AI Review", "AiReviewSettingsPanel", order: 50)
            // Review-board shortcuts. Surface "local" outranks the player's own ArrowLeft/ArrowRight
            // bindings while the board is mounted, and users can rebind them in Cove's keyboard settings.
            // "X" / "N" / "P" are the primary keys because the video page's player owns F (fullscreen)
            // and the arrow keys (seek/volume); the alternates work wherever the host leaves them free.
            .AddKeyboardAction("review-false-positive", "Mark false positive", ["X", "F"], [new("local")], description: "The tag is not present in the focused segment.", group: "AI Review", order: 10)
            .AddKeyboardAction("review-true-positive", "Mark true positive", ["T"], [new("local")], description: "The tag is present in the focused segment.", group: "AI Review", order: 20)
            .AddKeyboardAction("review-clear", "Clear verdict", ["U"], [new("local")], description: "Remove the verdict from the focused segment (restores a deleted segment first).", group: "AI Review", order: 30)
            .AddKeyboardAction("review-open", "Open segment", ["Enter"], [new("local")], description: "Open the page of the focused segment.", group: "AI Review", order: 40)
            .AddKeyboardAction("review-next", "Next segment", ["N", "ArrowRight"], [new("local")], group: "AI Review", order: 60, repeatable: true)
            .AddKeyboardAction("review-previous", "Previous segment", ["P", "ArrowLeft"], [new("local")], group: "AI Review", order: 70, repeatable: true)
            .AddKeyboardAction("review-down", "Segment below", ["ArrowDown"], [new("local")], group: "AI Review", order: 80, repeatable: true)
            .AddKeyboardAction("review-up", "Segment above", ["ArrowUp"], [new("local")], group: "AI Review", order: 90, repeatable: true)
            .Build();

    public IEnumerable<PermissionDefinition> ContributePermissions()
    {
        var source = $"extension:{Id}";
        return
        [
            new(ReviewPermission, "AI Review", "Review AI tag segments: record verdicts, export frames and delete false-positive segments from videos.", Dangerous: true, Source: source, GrantToAdminsByDefault: true),
        ];
    }

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        _dataRoot = Path.Combine(context.DataDirectory, ".ai-review-data");
        services.AddSingleton<AiReviewSettingsProvider>();
        services.AddSingleton(sp => new AiReviewFrameService(
            sp.GetRequiredService<IServiceScopeFactory>(),
            sp.GetService<CoveConfiguration>(),
            _dataRoot));
        services.AddSingleton(sp => new AiReviewExportService(
            sp.GetRequiredService<IServiceScopeFactory>(),
            sp.GetRequiredService<AiReviewSettingsProvider>(),
            sp.GetRequiredService<AiReviewFrameService>(),
            _dataRoot));
        services.AddSingleton<AiReviewQueryService>();
        services.AddSingleton<AiReviewMarkService>();
    }

    public override async Task InitializeAsync(IServiceProvider services, CancellationToken ct = default)
    {
        Directory.CreateDirectory(_dataRoot);
        var settingsProvider = services.GetRequiredService<AiReviewSettingsProvider>();
        settingsProvider.Attach(token => AiReviewSettingsStore.LoadAsync(Store, token));

        var exportService = services.GetRequiredService<AiReviewExportService>();
        exportService.Start();
        try
        {
            await exportService.EnqueuePendingAsync(includeFailed: false, ct);
        }
        catch
        {
            // The table may not exist yet on a brand-new install until migrations run; the next mark re-queues.
        }
    }

    public override Task ShutdownAsync(CancellationToken ct = default)
    {
        return Task.CompletedTask;
    }

    public override void ConfigureModel(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<ExtAiReviewMarkEntity>(entity =>
        {
            entity.ToTable("ext_ai_review_mark");
            entity.HasKey(item => item.Id);
            entity.Property(item => item.Id).HasColumnName("id");
            entity.Property(item => item.VideoId).HasColumnName("video_id");
            entity.Property(item => item.SegmentId).HasColumnName("segment_id");
            entity.Property(item => item.TagId).HasColumnName("tag_id");
            entity.Property(item => item.TagName).HasColumnName("tag_name");
            entity.Property(item => item.ModelKey).HasColumnName("model_key");
            entity.Property(item => item.SourceKey).HasColumnName("source_key");
            entity.Property(item => item.SourceRunId).HasColumnName("source_run_id");
            entity.Property(item => item.PayloadJson).HasColumnName("payload_json");
            entity.Property(item => item.StartSec).HasColumnName("start_sec");
            entity.Property(item => item.EndSec).HasColumnName("end_sec");
            entity.Property(item => item.Confidence).HasColumnName("confidence");
            entity.Property(item => item.Verdict).HasColumnName("verdict");
            entity.Property(item => item.ReviewerUserId).HasColumnName("reviewer_user_id");
            entity.Property(item => item.ReviewerName).HasColumnName("reviewer_name");
            entity.Property(item => item.Note).HasColumnName("note");
            entity.Property(item => item.FrameTimesJson).HasColumnName("frame_times_json");
            entity.Property(item => item.ExportStatus).HasColumnName("export_status");
            entity.Property(item => item.ExportedFilesJson).HasColumnName("exported_files_json");
            entity.Property(item => item.ExportError).HasColumnName("export_error");
            entity.Property(item => item.ExportedAt).HasColumnName("exported_at");
            entity.Property(item => item.SegmentDeletedAt).HasColumnName("segment_deleted_at");
            entity.Property(item => item.CreatedAt).HasColumnName("created_at");
            entity.Property(item => item.UpdatedAt).HasColumnName("updated_at");
            entity.HasIndex(item => item.SegmentId).IsUnique();
            entity.HasIndex(item => new { item.TagId, item.Verdict });
            entity.HasIndex(item => item.VideoId);
        });
    }

    protected override void DefineMigrations()
    {
        Migration("001_create_review_marks", """
            CREATE TABLE IF NOT EXISTS ext_ai_review_mark (
                id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                video_id integer NOT NULL,
                segment_id integer NULL,
                tag_id integer NOT NULL,
                tag_name text NOT NULL,
                model_key text NULL,
                source_key text NOT NULL,
                source_run_id text NULL,
                payload_json text NULL,
                start_sec double precision NOT NULL,
                end_sec double precision NOT NULL,
                confidence double precision NULL,
                verdict text NOT NULL,
                reviewer_user_id integer NULL,
                reviewer_name text NULL,
                note text NULL,
                frame_times_json text NOT NULL,
                export_status text NOT NULL,
                exported_files_json text NOT NULL,
                export_error text NULL,
                exported_at timestamp with time zone NULL,
                segment_deleted_at timestamp with time zone NULL,
                created_at timestamp with time zone NOT NULL,
                updated_at timestamp with time zone NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ix_ext_ai_review_mark_segment_id ON ext_ai_review_mark (segment_id);
            CREATE INDEX IF NOT EXISTS ix_ext_ai_review_mark_tag_id_verdict ON ext_ai_review_mark (tag_id, verdict);
            CREATE INDEX IF NOT EXISTS ix_ext_ai_review_mark_video_id ON ext_ai_review_mark (video_id);
            """);
    }

    public override void MapEndpoints(IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("/api/ext/ai-review").WithTags("AI.Review").RequireCovePermission(ReviewPermission);

        group.MapGet("/settings", async (AiReviewExportService exportService, CancellationToken ct) =>
        {
            var settings = await AiReviewSettingsStore.LoadAsync(Store, ct);
            return Results.Ok(new
            {
                settings,
                effectiveExportRootPath = exportService.ResolveExportRoot(settings),
                manifestPath = exportService.ManifestPath(settings),
            });
        });

        group.MapPut("/settings", async (AiReviewSettings settings, AiReviewSettingsProvider provider, AiReviewExportService exportService, CancellationToken ct) =>
        {
            try
            {
                var normalized = settings.Normalize();
                if (!string.IsNullOrWhiteSpace(normalized.ExportRootPath))
                {
                    Directory.CreateDirectory(normalized.ExportRootPath);
                }

                await AiReviewSettingsStore.SaveAsync(Store, normalized, ct);
                provider.Invalidate();
                return Results.Ok(new
                {
                    settings = normalized,
                    effectiveExportRootPath = exportService.ResolveExportRoot(normalized),
                    manifestPath = exportService.ManifestPath(normalized),
                });
            }
            catch (Exception ex)
            {
                return Results.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
            }
        });

        group.MapGet("/tags", async (int? videoId, AiReviewQueryService query, CancellationToken ct)
            => Results.Ok(await query.GetTagsAsync(videoId, ct)));

        group.MapGet("/videos/{videoId:int}/count", async (int videoId, AiReviewQueryService query, CancellationToken ct)
            => Results.Ok(new { count = await query.CountUnreviewedAsync(videoId, ct) }));

        group.MapPost("/queue", async (AiReviewQueueRequest request, AiReviewQueryService query, CancellationToken ct)
            => Results.Ok(await query.GetQueueAsync(request, ct)));

        group.MapPost("/marks", async (AiReviewSetVerdictRequest request, AiReviewMarkService marks, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            try
            {
                var item = await marks.SetVerdictAsync(request, principalAccessor.Current, ct);
                return item is null ? Results.NotFound(new { detail = "Segment not found." }) : Results.Ok(item);
            }
            catch (ArgumentException ex)
            {
                return Results.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
            }
        });

        group.MapPost("/marks/bulk", async (AiReviewBulkVerdictRequest request, AiReviewMarkService marks, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            try
            {
                var items = new List<AiReviewQueueItem>();
                foreach (var segmentId in request.SegmentIds.Distinct())
                {
                    var item = await marks.SetVerdictAsync(new AiReviewSetVerdictRequest(segmentId, request.Verdict, null), principalAccessor.Current, ct);
                    if (item is not null)
                    {
                        items.Add(item);
                    }
                }

                return Results.Ok(items);
            }
            catch (ArgumentException ex)
            {
                return Results.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
            }
        });

        group.MapPost("/marks/bulk-clear", async (AiReviewBulkClearRequest request, AiReviewMarkService marks, CancellationToken ct) =>
        {
            var cleared = 0;
            var skipped = 0;
            foreach (var markId in request.MarkIds.Distinct())
            {
                try
                {
                    if (await marks.ClearVerdictAsync(markId, ct))
                    {
                        cleared++;
                    }
                }
                catch (InvalidOperationException)
                {
                    skipped++;
                }
            }

            return Results.Ok(new { cleared, skipped });
        });

        group.MapDelete("/marks/{markId:int}", async (int markId, AiReviewMarkService marks, CancellationToken ct) =>
        {
            try
            {
                return await marks.ClearVerdictAsync(markId, ct) ? Results.NoContent() : Results.NotFound();
            }
            catch (InvalidOperationException ex)
            {
                return Results.Problem(ex.Message, statusCode: StatusCodes.Status409Conflict);
            }
        });

        group.MapPost("/marks/{markId:int}/restore", async (int markId, AiReviewMarkService marks, CancellationToken ct) =>
        {
            var item = await marks.RestoreAsync(markId, ct);
            return item is null ? Results.NotFound() : Results.Ok(item);
        });

        group.MapPost("/apply", async (AiReviewApplyRequest request, AiReviewMarkService marks, CancellationToken ct)
            => Results.Ok(await marks.ApplyAsync(request, ct)));

        group.MapGet("/export/status", async (AiReviewExportService exportService, CancellationToken ct)
            => Results.Ok(await exportService.GetStatusAsync(ct)));

        group.MapPost("/export/retry", async (AiReviewExportService exportService, CancellationToken ct) =>
        {
            var queued = await exportService.EnqueuePendingAsync(includeFailed: true, ct);
            exportService.RequestManifestRewrite();
            return Results.Ok(new { queued });
        });

        group.MapPost("/export/manifest", async (AiReviewExportService exportService, CancellationToken ct) =>
        {
            await exportService.RewriteManifestAsync(ct);
            var status = await exportService.GetStatusAsync(ct);
            return Results.Ok(new { manifestPath = status.ManifestPath });
        });
    }
}

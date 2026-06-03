using AI.Faces;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiFaceIdentityReconcilerTests
{
    [Fact]
    public void Reconcile_MergesSimilarProvisionalIdentitiesAndPromotesAcrossAssets()
    {
        var snapshot = new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 3,
            Identities =
            [
                new StoredFaceIdentity
                {
                    FaceKey = "face-0001",
                    LifecycleStatus = StoredFaceIdentityLifecycle.Provisional,
                    ObservationCount = 3,
                    AssetIds = ["video-a"],
                    Anchors =
                    [
                        new StoredFaceAnchor
                        {
                            ModelKey = "face_embedding_torchexport",
                            QualityScore = 20.0,
                            Vector = [1f, 0f],
                        },
                    ],
                },
                new StoredFaceIdentity
                {
                    FaceKey = "face-0002",
                    LifecycleStatus = StoredFaceIdentityLifecycle.Provisional,
                    ObservationCount = 2,
                    AssetIds = ["video-b"],
                    Anchors =
                    [
                        new StoredFaceAnchor
                        {
                            ModelKey = "face_embedding_torchexport",
                            QualityScore = 18.0,
                            Vector = [0.999f, 0.001f],
                        },
                    ],
                },
            ],
        };

        var report = new AiFaceIdentityReconciler().Reconcile(snapshot, referencePack: null, new AiFacesSettings());

        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal("face-0001", identity.FaceKey);
        Assert.Equal(StoredFaceIdentityLifecycle.Promoted, identity.LifecycleStatus);
        Assert.Equal("multi-asset", identity.PromotionReason);
        Assert.Equal(5, identity.ObservationCount);
        Assert.Equal(2, identity.AssetIds.Count);
        Assert.Equal(1, report.MergedIdentityCount);
        Assert.Equal(1, report.EvidencePromotedIdentityCount);
    }

    [Fact]
    public void Reconcile_DoesNotMergeDifferentPromotedUnknownIdentities()
    {
        var snapshot = new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 3,
            Identities =
            [
                CreatePromotedIdentity("face-0001", [1f, 0f]),
                CreatePromotedIdentity("face-0002", [0.999f, 0.001f]),
            ],
        };

        var report = new AiFaceIdentityReconciler().Reconcile(snapshot, referencePack: null, new AiFacesSettings());

        Assert.Equal(2, snapshot.Identities.Count);
        Assert.Equal(0, report.MergedIdentityCount);
    }

    [Fact]
    public void Reconcile_MergesHighConfidenceSameAssetPromotedUnknownFragments()
    {
        var snapshot = new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 3,
            Identities =
            [
                CreatePromotedIdentity("face-0001", [1f, 0f], "video-5634"),
                CreatePromotedIdentity("face-0002", [0.999f, 0.001f], "video-5634"),
            ],
        };

        var report = new AiFaceIdentityReconciler().Reconcile(snapshot, referencePack: null, new AiFacesSettings());

        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal("face-0001", identity.FaceKey);
        Assert.Equal(1, report.MergedIdentityCount);
        Assert.Equal("face-0001", report.MergedFaceKeyMap["face-0002"]);
    }

    [Fact]
    public void Reconcile_MergesSameAssetReferenceBackedFragmentAtIdentityThreshold()
    {
        var snapshot = new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 3,
            Identities =
            [
                CreatePromotedIdentity("face-0001", [1f, 0f], "video-5634", "ref-zazie", "Zazie Skymm", "reference"),
                CreatePromotedIdentity("face-0002", [0.48f, 0.8772685f], "video-5634"),
            ],
        };

        var report = new AiFaceIdentityReconciler().Reconcile(snapshot, referencePack: null, new AiFacesSettings());

        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal("face-0001", identity.FaceKey);
        Assert.Equal(1, report.MergedIdentityCount);
        Assert.Equal("face-0001", report.MergedFaceKeyMap["face-0002"]);
    }

    [Fact]
    public void Reconcile_MergesSameAssetAnonymousVideoEvidenceFragmentsWithoutReferencePack()
    {
        var snapshot = new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 4,
            Identities =
            [
                CreatePromotedIdentity("face-0001", [1f, 0f, 0f], "video-5634"),
                CreatePromotedIdentity("face-0002", [0.48f, 0.8772685f, 0f], "video-5634"),
                CreatePromotedIdentity("face-0003", [0.43f, 0.3517f, 0.8315f], "video-5634"),
            ],
        };

        var report = new AiFaceIdentityReconciler().Reconcile(snapshot, referencePack: null, new AiFacesSettings());

        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal("face-0001", identity.FaceKey);
        Assert.Equal(2, report.MergedIdentityCount);
        Assert.Equal("face-0001", report.MergedFaceKeyMap["face-0002"]);
        Assert.Equal("face-0001", report.MergedFaceKeyMap["face-0003"]);
    }

    [Fact]
    public void Reconcile_MergesSameAssetDuplicateReferenceIdentitiesBelowConsolidationThreshold()
    {
        var snapshot = new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 3,
            Identities =
            [
                CreatePromotedIdentity("face-0001", [1f, 0f], "video-5634", "ref-tiffany", "Tiffany Tatum", "reference"),
                CreatePromotedIdentity("face-0002", [0.61f, 0.792401f], "video-5634", "ref-tiffany", "Tiffany Tatum", "reference"),
            ],
        };

        var report = new AiFaceIdentityReconciler().Reconcile(snapshot, referencePack: null, new AiFacesSettings());

        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal("face-0001", identity.FaceKey);
        Assert.Equal(1, report.MergedIdentityCount);
        Assert.Equal("face-0001", report.MergedFaceKeyMap["face-0002"]);
    }

    private static StoredFaceIdentity CreatePromotedIdentity(
        string faceKey,
        IReadOnlyList<float> vector,
        string? assetId = null,
        string? referenceExternalId = null,
        string? label = null,
        string? promotionReason = null)
        => new()
        {
            FaceKey = faceKey,
            Label = label,
            LifecycleStatus = StoredFaceIdentityLifecycle.Promoted,
            PromotionReason = promotionReason ?? "video-evidence",
            ReferenceExternalId = referenceExternalId,
            ReferenceDisplayName = label,
            ObservationCount = 4,
            AssetIds = [assetId ?? $"asset-{faceKey}"],
            Anchors =
            [
                new StoredFaceAnchor
                {
                    ModelKey = "face_embedding_torchexport",
                    QualityScore = 20.0,
                    Vector = vector.ToList(),
                },
            ],
        };
}
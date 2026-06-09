using Cove.Core.DTOs;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

namespace AI.Faces;

// Singleton bridges that adapt AI.Faces' scoped face services onto the cross-extension service
// exchange. Since the extensions-runtime redesign each extension lives in its own DI container, and
// host code (Cove's FacesController, AiDataPurgeService) consumes these contracts as singletons read
// from the exchange. The host has no per-call DI scope to hand us, so each call opens its own scope
// and resolves the real scoped implementation (which needs a DbContext and scoped repositories)
// inside it. ICurrentPrincipalAccessor is a host singleton backed by AsyncLocal, so the calling
// request's principal still flows into the child scope.

internal sealed class ExtensionFaceSuggester(IServiceScopeFactory scopeFactory) : IFaceSuggester
{
    public async Task<IReadOnlyList<FaceSuggestionDto>> SuggestForAsync(int faceId, int maxResults, CancellationToken cancellationToken = default)
    {
        await using var scope = scopeFactory.CreateAsyncScope();
        return await scope.ServiceProvider.GetRequiredService<AiFaceSuggester>().SuggestForAsync(faceId, maxResults, cancellationToken);
    }

    public async Task<IReadOnlyList<FaceSuggestionDto>> SuggestForAsync(int faceId, int maxResults, FaceSuggestionOptions options, CancellationToken cancellationToken = default)
    {
        await using var scope = scopeFactory.CreateAsyncScope();
        return await scope.ServiceProvider.GetRequiredService<AiFaceSuggester>().SuggestForAsync(faceId, maxResults, options, cancellationToken);
    }

    public async Task<IReadOnlyDictionary<int, IReadOnlyList<FaceSuggestionDto>>> SuggestForBatchAsync(
        IReadOnlyCollection<int> faceIds,
        int maxResults,
        FaceSuggestionOptions options,
        CancellationToken cancellationToken = default)
    {
        await using var scope = scopeFactory.CreateAsyncScope();
        return await scope.ServiceProvider.GetRequiredService<AiFaceSuggester>().SuggestForBatchAsync(faceIds, maxResults, options, cancellationToken);
    }
}

internal sealed class ExtensionFaceSuggestionDecisionHandler(IServiceScopeFactory scopeFactory) : IFaceSuggestionDecisionHandler
{
    public async Task<FaceSuggestionDecisionOutcome> TryHandleAsync(FaceSuggestionDecisionRequest request, CancellationToken cancellationToken = default)
    {
        await using var scope = scopeFactory.CreateAsyncScope();
        return await scope.ServiceProvider.GetRequiredService<AiFaceReferenceSuggestionDecisionHandler>().TryHandleAsync(request, cancellationToken);
    }
}

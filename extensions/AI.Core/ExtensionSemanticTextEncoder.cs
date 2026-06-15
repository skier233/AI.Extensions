using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

using Pgvector;

namespace AI.Core;

// Singleton bridge that adapts AI.Core's scoped AiCoreSemanticTextEncoder onto the cross-extension
// service exchange. Since the extensions-runtime redesign each extension lives in its own DI
// container, and host/sibling code (notably AI.Visual's semantic search, which resolves encoders
// through the host ITextEncoderRegistry) consumes ITextEncoder as a singleton read from the
// exchange. The consumer has no per-call DI scope to hand us, so each call opens its own scope and
// resolves the real scoped encoder (which needs IExtensionStoreFactory and the typed
// NsfwAiServerClient) inside it. Without this bridge + the InitializeAsync PublishContributions
// call, the registry resolves no encoder for "semantic.v1" and the query is never sent to
// nsfw_ai_server, so visual semantic search silently returns no results.
internal sealed class ExtensionSemanticTextEncoder(IServiceScopeFactory scopeFactory) : ITextEncoder
{
    public string KindFamily => AiCoreSemanticTextEncoder.SemanticKindFamily;

    public async Task<Vector> EncodeAsync(string text, CancellationToken cancellationToken = default)
    {
        await using var scope = scopeFactory.CreateAsyncScope();
        return await scope.ServiceProvider
            .GetRequiredService<AiCoreSemanticTextEncoder>()
            .EncodeAsync(text, cancellationToken);
    }
}

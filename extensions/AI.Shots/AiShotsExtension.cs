using AI.Extensions.Abstractions;

using Cove.Plugins;
using Cove.Sdk;

using Microsoft.Extensions.DependencyInjection;

namespace AI.Shots;

public sealed class AiShotsExtension : CoveExtensionBase
{
    // Id, Name, Version and other metadata are sourced from extension.json by CoveExtensionBase.

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        services.AddScoped<IAiShotsFileLookup, VideoRepositoryFileLookup>();
        services.AddSingleton<AiShotsPersistenceService>();
        services.AddSingleton<IAiCapabilityContributor, AiShotsContributor>();
    }

    public override Task InitializeAsync(IServiceProvider services, CancellationToken ct = default)
    {
        PublishContributions<IAiCapabilityContributor>(services);
        return Task.CompletedTask;
    }
}

using System.Reflection;
using System.Runtime.Loader;

using AI.Core;
using AI.Extensions.Abstractions;

using Xunit;

namespace AI.Extensions.Tests;

// The host loads one copy of AI.Extensions.Abstractions for every extension, and disables an extension any of whose
// types fails to load. The copy that wins can predate contributor planning, so AI Core must load without the planning
// types and then plan every claim from run history.
public sealed class AiCoreSharedAbstractionsTests
{
    private static readonly Type[] PlanningTypes =
    [
        typeof(IAiClaimPlanningContributor),
        typeof(AiClaimPlanningTarget),
        typeof(AiClaimPlanningDecision),
        typeof(AiClaimPlanningVerdict),
        typeof(AiDispatchMetadataKeys),
        typeof(AiDispatchMetadata),
    ];

    [Fact]
    public void AiCoreLoadsAgainstAbstractionsThatPredatePlanning()
    {
        var prePlanningPath = Path.Combine(AppContext.BaseDirectory, "pre-planning", "AI.Extensions.Abstractions.dll");
        Assert.True(File.Exists(prePlanningPath), $"Missing {prePlanningPath}; it is built by the PrePlanningAbstractions project.");

        var context = new PrePlanningLoadContext(prePlanningPath);
        try
        {
            var aiCore = context.LoadFromAssemblyPath(typeof(AiCoreOrchestrator).Assembly.Location);
            var abstractions = context.LoadFromAssemblyName(new AssemblyName("AI.Extensions.Abstractions"));
            Assert.Equal(prePlanningPath, abstractions.Location);
            Assert.Null(abstractions.GetType("AI.Extensions.Abstractions.IAiClaimPlanningContributor"));

            // Throws ReflectionTypeLoadException naming the type if any AI Core type needs a planning type to load.
            var types = aiCore.GetTypes();

            var orchestrator = Assert.Single(types, static type => type.FullName == "AI.Core.AiCoreOrchestrator");
            var isPlanningContributor = orchestrator.GetMethod("IsPlanningContributor", BindingFlags.NonPublic | BindingFlags.Static)!;
            Assert.Equal(false, isPlanningContributor.Invoke(null, [null]));
        }
        finally
        {
            context.Unload();
        }
    }

    [Fact]
    public void NoAiCoreTypeHoldsAPlanningType()
    {
        // A compiler-generated async state machine, closure or lambda cache that holds a planning type in a field
        // fails to load without it, taking AI Core down; only plain method bodies may name these types.
        var offenders = new List<string>();
        foreach (var type in typeof(AiCoreOrchestrator).Assembly.GetTypes())
        {
            if (type.BaseType is { } baseType && Mentions(baseType))
            {
                offenders.Add($"{type.FullName} derives from {baseType}");
            }

            offenders.AddRange(type.GetInterfaces().Where(Mentions).Select(iface => $"{type.FullName} implements {iface}"));
            offenders.AddRange(type
                .GetFields(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                .Where(field => Mentions(field.FieldType))
                .Select(field => $"{type.FullName}.{field.Name}: {field.FieldType}"));
        }

        Assert.Empty(offenders);
    }

    private static bool Mentions(Type type)
        => PlanningTypes.Contains(type)
            || (type.HasElementType && Mentions(type.GetElementType()!))
            || (type.IsGenericType && type.GetGenericArguments().Any(Mentions));

    private sealed class PrePlanningLoadContext(string abstractionsPath) : AssemblyLoadContext("pre-planning-abstractions", isCollectible: true)
    {
        protected override Assembly? Load(AssemblyName assemblyName)
            => assemblyName.Name == "AI.Extensions.Abstractions" ? LoadFromAssemblyPath(abstractionsPath) : null;
    }
}

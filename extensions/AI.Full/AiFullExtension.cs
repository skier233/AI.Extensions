using Cove.Plugins;
using Cove.Sdk;

namespace AI.Full;

public sealed class AiFullExtension : CoveExtensionBase
{
    public override string Id => "cove.community.ai.full";

    public override string Name => "AI Full";

    public override string Version => "0.1.0";

    public override string Description => "Dependency bundle for the full Cove AI extension family.";

    public override string Author => "Cove Team";

    public override string Url => "https://github.com/yourcove/AI.Extensions";

    public override string MinCoveVersion => "0.0.32";

    public override IReadOnlyList<string> Categories =>
    [
        ExtensionCategories.Tools,
        ExtensionCategories.Automation,
        "ai",
        "bundle",
    ];

    public override IReadOnlyDictionary<string, string> Dependencies => new Dictionary<string, string>
    {
        ["cove.community.ai.core"] = ">=0.1.0",
        ["cove.community.ai.tagging"] = ">=0.1.0",
        ["cove.community.ai.faces"] = ">=0.1.0",
        ["cove.community.ai.visual"] = ">=0.1.0",
        ["cove.community.ai.audio"] = ">=0.1.0",
    };
}

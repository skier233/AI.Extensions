using Cove.Plugins;
using Cove.Sdk;

namespace AI.Full;

// AI Full is a dependency-only bundle (extension.json "kind": "bundle"); the host treats it as
// manifest-only and never instantiates this class. All metadata — including the dependency set that
// pulls in the rest of the AI family — lives solely in extension.json.
public sealed class AiFullExtension : CoveExtensionBase
{
}

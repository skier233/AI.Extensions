using System.Runtime.CompilerServices;

using AI.Faces;

using Cove.Data;

namespace AI.Extensions.Tests;

internal static class FaceIdentityModelInitializer
{
    // The AI.Faces identity-graph tables (ext_ai_faces_identity[_anchor]) are contributed to CoveContext's
    // model through AiFacesExtension.ConfigureModel rather than declared on CoveContext itself. CoveContext
    // builds (and globally caches) its model from the static data-extension list, so register that
    // contribution once — before any in-memory CoveContext model is built — so tests that exercise
    // DbFaceIdentityStore can query those tables. AiFacesExtension has no constructor side effects beyond
    // wiring its own (here unused) job/event tables, so constructing one purely for ConfigureModel is safe.
    [ModuleInitializer]
    internal static void Init()
        => CoveContext.SetDataExtensions([new AiFacesExtension()]);
}

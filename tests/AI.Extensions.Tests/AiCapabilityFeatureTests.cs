using System.Text.Json;

using AI.Extensions.Abstractions;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiCapabilityFeatureTests
{
    [Fact]
    public void FeaturesAreSelectedByDefaultUnlessTheySayOtherwise()
    {
        Assert.True(new AiCapabilityFeature("tagging", "Tagging", ["tagging.video.frame"]).SelectedByDefault);
        Assert.False(new AiCapabilityFeature("shots", "Shot Boundaries", ["shots.video.asset"]) { SelectedByDefault = false }.SelectedByDefault);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheRunAiDialogReadsSelectedByDefaultFromTheCapabilitiesResponse(bool selectedByDefault)
    {
        // The capabilities endpoint serializes descriptors with the web defaults; the dialog reads selectedByDefault.
        var feature = new AiCapabilityFeature("shots", "Shot Boundaries", ["shots.video.asset"]) { SelectedByDefault = selectedByDefault };

        var json = JsonSerializer.SerializeToElement(feature, new JsonSerializerOptions(JsonSerializerDefaults.Web));

        Assert.Equal(selectedByDefault, json.GetProperty("selectedByDefault").GetBoolean());
    }
}

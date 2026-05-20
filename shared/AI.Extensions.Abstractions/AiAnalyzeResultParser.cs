using System.Text.Json;

namespace AI.Extensions.Abstractions;

public static class AiAnalyzeResultParser
{
    public static AiAnalyzeResult Parse(string mediaKind, JsonElement payload)
    {
        return mediaKind switch
        {
            AiMediaKinds.Image => ParseImage(payload),
            AiMediaKinds.Video => ParseVideo(payload),
            AiMediaKinds.Audio => ParseAudio(payload),
            _ => throw new ArgumentOutOfRangeException(nameof(mediaKind), mediaKind, "Unsupported media kind."),
        };
    }

    private static AiAnalyzeResult ParseImage(JsonElement payload)
    {
        return new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Image,
            AssetId = GetString(payload, "asset_id") ?? string.Empty,
            Models = ParseModels(payload),
            RequestedModelNames = GetStringArray(payload, "requested_model_names"),
            AssetAnalysis = payload.TryGetProperty("analysis", out var analysis)
                ? ParseAnalysisNode(analysis)
                : new AiAnalysisNode(),
            Metrics = ParseMetrics(payload),
        };
    }

    private static AiAnalyzeResult ParseVideo(JsonElement payload)
    {
        return new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = GetString(payload, "asset_id") ?? string.Empty,
            DurationSeconds = GetDouble(payload, "duration_seconds"),
            FrameIntervalSeconds = GetDouble(payload, "frame_interval_seconds") ?? GetDouble(payload, "frame_interval"),
            Models = ParseModels(payload),
            RequestedModelNames = GetStringArray(payload, "requested_model_names"),
            Frames = payload.TryGetProperty("frames", out var frames)
                ? ParseTemporalSlices(frames, "frame")
                : [],
            Metrics = ParseMetrics(payload),
        };
    }

    private static AiAnalyzeResult ParseAudio(JsonElement payload)
    {
        return new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Audio,
            AssetId = GetString(payload, "asset_id") ?? string.Empty,
            Models = ParseModels(payload),
            RequestedModelNames = GetStringArray(payload, "requested_model_names"),
            Windows = payload.TryGetProperty("windows", out var windows)
                ? ParseTemporalSlices(windows, "window")
                : [],
            Metrics = ParseMetrics(payload),
        };
    }

    private static IReadOnlyList<AiModelDescriptor> ParseModels(JsonElement payload)
    {
        if (!payload.TryGetProperty("models", out var models) || models.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var parsed = new List<AiModelDescriptor>();
        foreach (var model in models.EnumerateArray())
        {
            parsed.Add(new AiModelDescriptor(
                GetString(model, "config_name") ?? GetString(model, "name") ?? string.Empty,
                GetString(model, "name") ?? string.Empty,
                GetString(model, "type"),
                GetStringArray(model, "capabilities"),
                GetStringArray(model, "supported_scopes"),
                GetStringArray(model, "categories"),
                GetString(model, "version"),
                GetBool(model, "active"),
                GetBool(model, "loaded"),
                GetBool(model, "pinned")));
        }

        return parsed;
    }

    private static IReadOnlyList<AiTemporalSlice> ParseTemporalSlices(JsonElement slices, string sliceKind)
    {
        if (slices.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var parsed = new List<AiTemporalSlice>();
        foreach (var slice in slices.EnumerateArray())
        {
            var analysis = slice.TryGetProperty("analysis", out var analysisElement)
                ? ParseAnalysisNode(analysisElement)
                : new AiAnalysisNode();
            parsed.Add(new AiTemporalSlice(
                sliceKind,
                GetInt(slice, "index"),
                GetDouble(slice, "time_seconds"),
                GetDouble(slice, "start"),
                GetDouble(slice, "end"),
                analysis));
        }

        return parsed;
    }

    private static AiAnalysisNode ParseAnalysisNode(JsonElement analysis)
    {
        var tags = new List<AiTagPrediction>();
        var classifications = new List<AiClassificationPrediction>();
        var detections = new List<AiDetectionObservation>();
        var embeddings = new List<AiEmbeddingObservation>();
        var branches = new List<AiRegionBranch>();
        var other = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (analysis.ValueKind != JsonValueKind.Object)
        {
            return new AiAnalysisNode();
        }

        if (analysis.TryGetProperty("capabilities", out var capabilities) && capabilities.ValueKind == JsonValueKind.Object)
        {
            if (capabilities.TryGetProperty("tagging", out var tagging))
            {
                tags.AddRange(ParseLabelCollection<AiTagPrediction>(tagging, static (modelKey, label, confidence) => new AiTagPrediction(modelKey, label, confidence)));
            }

            if (capabilities.TryGetProperty("classification", out var classification))
            {
                classifications.AddRange(ParseLabelCollection<AiClassificationPrediction>(classification, static (modelKey, label, confidence) => new AiClassificationPrediction(modelKey, label, confidence)));
            }

            if (capabilities.TryGetProperty("detection", out var detection))
            {
                detections.AddRange(ParseDetections(detection));
            }

            if (capabilities.TryGetProperty("embedding", out var embedding))
            {
                embeddings.AddRange(ParseEmbeddings(embedding, null, null));
            }
        }

        if (analysis.TryGetProperty("region_branches", out var regionBranches) && regionBranches.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in regionBranches.EnumerateObject())
            {
                if (property.Value.ValueKind != JsonValueKind.Array)
                {
                    continue;
                }

                foreach (var branch in property.Value.EnumerateArray())
                {
                    var branchDetectionIndex = GetInt(branch, "detection_index");
                    if (!branchDetectionIndex.HasValue &&
                        branch.TryGetProperty("other", out var branchOther) &&
                        branchOther.ValueKind == JsonValueKind.Object)
                    {
                        branchDetectionIndex = GetInt(branchOther, "detection_index");
                    }

                    branches.Add(new AiRegionBranch(
                        property.Name,
                        branchDetectionIndex,
                        ParseAnalysisNode(branch)));
                }
            }
        }

        if (analysis.TryGetProperty("other", out var otherBlock) && otherBlock.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in otherBlock.EnumerateObject())
            {
                other[property.Name] = property.Value.ToString();
            }
        }

        return new AiAnalysisNode
        {
            Tags = tags,
            Classifications = classifications,
            Detections = detections,
            Embeddings = embeddings,
            RegionBranches = branches,
            Other = other,
        };
    }

    private static IReadOnlyList<TPrediction> ParseLabelCollection<TPrediction>(
        JsonElement block,
        Func<string, string, double?, TPrediction> factory)
    {
        if (block.ValueKind != JsonValueKind.Object)
        {
            return [];
        }

        var predictions = new List<TPrediction>();
        foreach (var modelProperty in block.EnumerateObject())
        {
            if (modelProperty.Value.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            foreach (var item in modelProperty.Value.EnumerateArray())
            {
                switch (item.ValueKind)
                {
                    case JsonValueKind.String:
                        predictions.Add(factory(modelProperty.Name, item.GetString() ?? string.Empty, null));
                        break;

                    case JsonValueKind.Array:
                    {
                        using var enumerator = item.EnumerateArray();
                        if (!enumerator.MoveNext())
                        {
                            break;
                        }

                        var label = enumerator.Current.ValueKind == JsonValueKind.String
                            ? enumerator.Current.GetString() ?? string.Empty
                            : enumerator.Current.ToString();
                        double? confidence = null;
                        if (enumerator.MoveNext() && TryReadDouble(enumerator.Current, out var score))
                        {
                            confidence = score;
                        }

                        predictions.Add(factory(modelProperty.Name, label, confidence));
                        break;
                    }

                    case JsonValueKind.Object:
                    {
                        var label = GetString(item, "tag")
                            ?? GetString(item, "label")
                            ?? GetString(item, "name")
                            ?? item.ToString();
                        var confidence = GetDouble(item, "confidence")
                            ?? GetDouble(item, "score")
                            ?? GetDouble(item, "probability");
                        predictions.Add(factory(modelProperty.Name, label, confidence));
                        break;
                    }
                }
            }
        }

        return predictions;
    }

    private static IReadOnlyList<AiDetectionObservation> ParseDetections(JsonElement block)
    {
        if (block.ValueKind != JsonValueKind.Object)
        {
            return [];
        }

        var detections = new List<AiDetectionObservation>();
        foreach (var modelProperty in block.EnumerateObject())
        {
            if (modelProperty.Value.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            var detectionIndex = 0;
            foreach (var item in modelProperty.Value.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Object)
                {
                    detectionIndex++;
                    continue;
                }

                if (!TryParseBoundingBox(item, out var boundingBox))
                {
                    detectionIndex++;
                    continue;
                }

                var label = GetString(item, "class")
                    ?? GetString(item, "label")
                    ?? GetString(item, "name")
                    ?? modelProperty.Name;
                var score = GetDouble(item, "score") ?? GetDouble(item, "confidence") ?? 0.0;
                detections.Add(new AiDetectionObservation(
                    modelProperty.Name,
                    detectionIndex,
                    label,
                    score,
                    boundingBox,
                    ParseMetadata(item, ["bbox", "score", "confidence", "class", "label", "name"])));
                detectionIndex++;
            }
        }

        return detections;
    }

    private static IReadOnlyList<AiEmbeddingObservation> ParseEmbeddings(JsonElement block, int? branchDetectionIndex, string? branchKey)
    {
        if (block.ValueKind != JsonValueKind.Object)
        {
            return [];
        }

        var embeddings = new List<AiEmbeddingObservation>();
        foreach (var modelProperty in block.EnumerateObject())
        {
            if (modelProperty.Value.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            foreach (var item in modelProperty.Value.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }

                var vector = GetFloatArray(item, "vector");
                if (vector.Count == 0)
                {
                    continue;
                }

                embeddings.Add(new AiEmbeddingObservation(
                    modelProperty.Name,
                    GetString(item, "scope") ?? (branchDetectionIndex.HasValue ? "region" : "asset"),
                    vector,
                    GetDouble(item, "norm"),
                    GetInt(item, "det_id") ?? GetInt(item, "detection_index") ?? branchDetectionIndex,
                    branchKey,
                    ParseMetadata(item, ["vector", "norm", "scope", "det_id", "detection_index"])));
            }
        }

        return embeddings;
    }

    private static IReadOnlyDictionary<string, string> ParseMetadata(JsonElement item, IReadOnlyCollection<string> excludedKeys)
    {
        var metadata = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (item.ValueKind != JsonValueKind.Object)
        {
            return metadata;
        }

        foreach (var property in item.EnumerateObject())
        {
            if (excludedKeys.Contains(property.Name))
            {
                continue;
            }

            metadata[property.Name] = property.Value.ToString();
        }

        return metadata;
    }

    private static IReadOnlyDictionary<string, double> ParseMetrics(JsonElement payload)
    {
        if (!payload.TryGetProperty("metrics", out var metrics) || metrics.ValueKind != JsonValueKind.Object)
        {
            return new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        }

        var parsed = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        foreach (var property in metrics.EnumerateObject())
        {
            if (TryReadDouble(property.Value, out var value))
            {
                parsed[property.Name] = value;
            }
        }

        return parsed;
    }

    private static bool TryParseBoundingBox(JsonElement item, out AiBoundingBox boundingBox)
    {
        boundingBox = default;
        if (item.TryGetProperty("bbox", out var bbox) && bbox.ValueKind == JsonValueKind.Array)
        {
            var values = bbox.EnumerateArray()
                .Select(static value => TryReadDouble(value, out var parsedValue) ? parsedValue : (double?)null)
                .Where(static value => value.HasValue)
                .Select(static value => value!.Value)
                .ToArray();
            if (values.Length == 4)
            {
                boundingBox = new AiBoundingBox(values[0], values[1], values[2], values[3]);
                return true;
            }
        }

        var x = GetDouble(item, "x");
        var y = GetDouble(item, "y");
        var width = GetDouble(item, "w") ?? GetDouble(item, "width");
        var height = GetDouble(item, "h") ?? GetDouble(item, "height");
        if (x.HasValue && y.HasValue && width.HasValue && height.HasValue)
        {
            boundingBox = new AiBoundingBox(x.Value, y.Value, x.Value + width.Value, y.Value + height.Value);
            return true;
        }

        return false;
    }

    private static string? GetString(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;

    private static double? GetDouble(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && TryReadDouble(property, out var value)
            ? value
            : null;

    private static int? GetInt(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) && TryReadInt(property, out var value)
            ? value
            : null;

    private static bool GetBool(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out var property) &&
           (property.ValueKind == JsonValueKind.True || property.ValueKind == JsonValueKind.False) &&
           property.GetBoolean();

    private static IReadOnlyList<string> GetStringArray(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var property) || property.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        return property.EnumerateArray()
            .Where(static item => item.ValueKind == JsonValueKind.String)
            .Select(static item => item.GetString() ?? string.Empty)
            .Where(static item => !string.IsNullOrWhiteSpace(item))
            .ToArray();
    }

    private static IReadOnlyList<float> GetFloatArray(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var property) || property.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var values = new List<float>();
        foreach (var item in property.EnumerateArray())
        {
            if (TryReadDouble(item, out var doubleValue))
            {
                values.Add((float)doubleValue);
            }
        }

        return values;
    }

    private static bool TryReadDouble(JsonElement element, out double value)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Number:
                return element.TryGetDouble(out value);

            case JsonValueKind.String:
                return double.TryParse(
                    element.GetString(),
                    System.Globalization.NumberStyles.Float | System.Globalization.NumberStyles.AllowThousands,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out value);

            default:
                value = default;
                return false;
        }
    }

    private static bool TryReadInt(JsonElement element, out int value)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Number:
                return element.TryGetInt32(out value);

            case JsonValueKind.String:
                return int.TryParse(
                    element.GetString(),
                    System.Globalization.NumberStyles.Integer,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out value);

            default:
                value = default;
                return false;
        }
    }
}
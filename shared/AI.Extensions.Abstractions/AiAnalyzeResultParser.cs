using System.Text.Json;

namespace AI.Extensions.Abstractions;

public static class AiAnalyzeResultParser
{
    private static readonly int[] AudioClassWhitelistIndices = [8, 9, 14, 22, 24, 25, 38, 39, 44, 45, 46];
    private static readonly int[] AudioClassSpeechIndices = [0, 1, 2, 3, 4, 5, 15];
    private static readonly int[] AudioClassMusicIndices = [27, 28, 32, 33, 34, 137, 138, 140, 141, 142, 143, 144, 145, 146, 153, 154, 162, 163, 164, 165, 166, 167, 168, 184, 266, 270];
    private static readonly int[] AudioClassMoanIndices = [25, 38, 39, 22, 24, 44, 45, 46, 14, 8, 9];
    private static readonly int[] AudioClassBreathIndices = [41, 26, 43];
    private const double AudioClassThreshold = 0.01;

    public static AiAnalyzeResult Parse(string mediaKind, JsonElement payload)
    {
        var content = GetPrimaryPayload(payload);
        return mediaKind switch
        {
            AiMediaKinds.Image => ParseImage(payload, content),
            AiMediaKinds.Video => ParseVideo(payload, content),
            AiMediaKinds.Audio => ParseAudio(payload, content),
            _ => throw new ArgumentOutOfRangeException(nameof(mediaKind), mediaKind, "Unsupported media kind."),
        };
    }

    private static AiAnalyzeResult ParseImage(JsonElement payload, JsonElement content)
    {
        return new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Image,
            AssetId = GetString(content, "asset_id") ?? GetString(payload, "asset_id") ?? string.Empty,
            Models = ParseModelsWithFallback(payload, content),
            RequestedModelNames = GetStringArrayWithFallback(payload, content, "requested_model_names"),
            AssetAnalysis = content.TryGetProperty("analysis", out var analysis)
                ? ParseAnalysisNode(analysis)
                : new AiAnalysisNode(),
            Metrics = ParseMetricsWithFallback(payload, content),
        };
    }

    private static AiAnalyzeResult ParseVideo(JsonElement payload, JsonElement content)
    {
        return new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = GetString(content, "asset_id") ?? GetString(payload, "asset_id") ?? string.Empty,
            DurationSeconds = GetDouble(content, "duration_seconds") ?? GetDouble(payload, "duration_seconds"),
            FrameIntervalSeconds = GetDouble(content, "frame_interval_seconds")
                ?? GetDouble(content, "frame_interval")
                ?? GetDouble(payload, "frame_interval_seconds")
                ?? GetDouble(payload, "frame_interval"),
            Models = ParseModelsWithFallback(payload, content),
            RequestedModelNames = GetStringArrayWithFallback(payload, content, "requested_model_names"),
            Frames = content.TryGetProperty("frames", out var frames)
                ? ParseTemporalSlices(frames, "frame")
                : [],
            Metrics = ParseMetricsWithFallback(payload, content),
        };
    }

    private static AiAnalyzeResult ParseAudio(JsonElement payload, JsonElement content)
    {
        return new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Audio,
            AssetId = GetString(content, "asset_id") ?? GetString(payload, "asset_id") ?? string.Empty,
            Models = ParseModelsWithFallback(payload, content),
            RequestedModelNames = GetStringArrayWithFallback(payload, content, "requested_model_names"),
            Windows = content.TryGetProperty("windows", out var windows)
                ? ParseTemporalSlices(windows, "window")
                : [],
            Metrics = ParseMetricsWithFallback(payload, content),
        };
    }

    private static JsonElement GetPrimaryPayload(JsonElement payload)
    {
        if (payload.ValueKind != JsonValueKind.Object || !payload.TryGetProperty("result", out var result))
        {
            return payload;
        }

        if (result.ValueKind == JsonValueKind.Object)
        {
            return result;
        }

        if (result.ValueKind == JsonValueKind.Array)
        {
            using var enumerator = result.EnumerateArray();
            if (enumerator.MoveNext())
            {
                return enumerator.Current;
            }
        }

        return payload;
    }

    private static IReadOnlyList<AiModelDescriptor> ParseModelsWithFallback(JsonElement payload, JsonElement content)
    {
        var models = ParseModels(content);
        return models.Count > 0 ? models : ParseModels(payload);
    }

    private static IReadOnlyList<string> GetStringArrayWithFallback(JsonElement payload, JsonElement content, string propertyName)
    {
        var values = GetStringArray(content, propertyName);
        return values.Count > 0 ? values : GetStringArray(payload, propertyName);
    }

    private static IReadOnlyDictionary<string, double> ParseMetricsWithFallback(JsonElement payload, JsonElement content)
    {
        var metrics = ParseMetrics(payload);
        return metrics.Count > 0 ? metrics : ParseMetrics(content);
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
                GetBool(model, "loaded")));
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
            if (sliceKind == "window")
            {
                analysis = AddAudioWindowClassification(slice, analysis);
            }
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
                embeddings.AddRange(ParseEmbeddingArray(property.Name, property.Value, null, null));
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
                            : ExtractLabel(enumerator.Current);
                        double? confidence = null;
                        if (enumerator.MoveNext() && TryReadDouble(enumerator.Current, out var score))
                        {
                            confidence = score;
                        }

                        if (!string.IsNullOrWhiteSpace(label))
                        {
                            predictions.Add(factory(modelProperty.Name, label, confidence));
                        }
                        break;
                    }

                    case JsonValueKind.Object:
                    {
                        var label = TryExtractAudioClassLabel(modelProperty.Name, item)
                            ?? GetString(item, "tag")
                            ?? GetString(item, "label")
                            ?? GetString(item, "name")
                            ?? GetString(item, "class")
                            ?? GetString(item, "class_name")
                            ?? GetString(item, "top_class")
                            ?? GetString(item, "top_label")
                            ?? ExtractLabel(item);
                        var confidence = GetDouble(item, "confidence")
                            ?? GetDouble(item, "score")
                            ?? GetDouble(item, "probability");
                        if (!string.IsNullOrWhiteSpace(label))
                        {
                            predictions.Add(factory(modelProperty.Name, label, confidence));
                        }
                        break;
                    }
                }
            }
        }

        return predictions;
    }

    private static string? TryExtractAudioClassLabel(string modelKey, JsonElement item)
    {
        if (!string.Equals(modelKey, "audio_classification_audioclass", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(GetString(item, "classifier"), "audioclass", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        var probabilities = GetFloatArray(item, "probabilities");
        if (probabilities.Count == 0)
        {
            return null;
        }

        var musicScore = MaxAudioClassScore(probabilities, AudioClassMusicIndices);
        var whitelistScore = MaxAudioClassScore(probabilities, AudioClassWhitelistIndices);
        var speechScore = MaxAudioClassScore(probabilities, AudioClassSpeechIndices);

        if (musicScore > 0.05 && musicScore > Math.Max(whitelistScore, speechScore))
        {
            return null;
        }

        if (Math.Max(whitelistScore, speechScore) < AudioClassThreshold)
        {
            return null;
        }

        var moanScore = MaxAudioClassScore(probabilities, AudioClassMoanIndices);
        var breathScore = MaxAudioClassScore(probabilities, AudioClassBreathIndices);
        return moanScore >= speechScore && moanScore >= breathScore
            ? "moan"
            : breathScore >= speechScore
                ? "breath"
                : "speech";
    }

    private static double MaxAudioClassScore(IReadOnlyList<float> probabilities, IReadOnlyList<int> indices)
    {
        var max = 0d;
        foreach (var index in indices)
        {
            if (index < probabilities.Count && probabilities[index] > max)
            {
                max = probabilities[index];
            }
        }

        return max;
    }

    private static AiAnalysisNode AddAudioWindowClassification(JsonElement slice, AiAnalysisNode analysis)
    {
        var dominantType = GetString(slice, "dominant_type");
        if (string.IsNullOrWhiteSpace(dominantType) || string.Equals(dominantType, "unknown", StringComparison.OrdinalIgnoreCase))
        {
            return analysis;
        }

        return new AiAnalysisNode
        {
            Tags = analysis.Tags,
            Classifications = analysis.Classifications
                .Concat([new AiClassificationPrediction("audio.summary", dominantType, null)])
                .ToArray(),
            Detections = analysis.Detections,
            Embeddings = analysis.Embeddings,
            RegionBranches = analysis.RegionBranches,
            Other = analysis.Other,
        };
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
            embeddings.AddRange(ParseEmbeddingArray(modelProperty.Name, modelProperty.Value, branchDetectionIndex, branchKey));
        }

        return embeddings;
    }

    private static IReadOnlyList<AiEmbeddingObservation> ParseEmbeddingArray(string modelKey, JsonElement items, int? branchDetectionIndex, string? branchKey)
    {
        if (items.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var embeddings = new List<AiEmbeddingObservation>();
        foreach (var item in items.EnumerateArray())
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
                modelKey,
                GetString(item, "scope") ?? (branchDetectionIndex.HasValue ? "region" : "asset"),
                vector,
                GetDouble(item, "norm"),
                GetInt(item, "det_id") ?? GetInt(item, "detection_index") ?? branchDetectionIndex,
                branchKey,
                ParseMetadata(item, ["vector", "norm", "scope", "det_id", "detection_index"])));
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

    private static string ExtractLabel(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            return element.ToString();
        }

        // Try to find a top-class / top-label field first
        if (element.TryGetProperty("top_class", out var topClass) && topClass.ValueKind == JsonValueKind.String)
        {
            return topClass.GetString() ?? string.Empty;
        }
        if (element.TryGetProperty("top_label", out var topLabel) && topLabel.ValueKind == JsonValueKind.String)
        {
            return topLabel.GetString() ?? string.Empty;
        }

        if (element.TryGetProperty("top5", out var top5))
        {
            var top5Label = ExtractTop5Label(top5);
            if (!string.IsNullOrWhiteSpace(top5Label))
                return top5Label;
        }

        // Try to find a probability / score distribution object and extract the top label
        if (element.TryGetProperty("probabilities", out var probabilities) && probabilities.ValueKind == JsonValueKind.Object)
        {
            var extractedTop = ExtractTopLabelFromScores(probabilities);
            if (!string.IsNullOrWhiteSpace(extractedTop))
                return extractedTop;
        }
        if (element.TryGetProperty("scores", out var scores) && scores.ValueKind == JsonValueKind.Object)
        {
            var extractedTop = ExtractTopLabelFromScores(scores);
            if (!string.IsNullOrWhiteSpace(extractedTop))
                return extractedTop;
        }

        // Generic fallback: if any nested object looks like a score map, use its top label.
        foreach (var property in element.EnumerateObject())
        {
            if (property.Value.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            var nestedTop = ExtractTopLabelFromScores(property.Value);
            if (!string.IsNullOrWhiteSpace(nestedTop))
            {
                return nestedTop;
            }
        }

        // Look for any direct string property that is a label, not model metadata.
        foreach (var property in element.EnumerateObject())
        {
            if (property.Value.ValueKind == JsonValueKind.String &&
                !string.Equals(property.Name, "classifier", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(property.Name, "model", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(property.Name, "modelKey", StringComparison.OrdinalIgnoreCase))
            {
                return property.Value.GetString() ?? string.Empty;
            }
        }

        // Look for a direct numeric property (property name is the label)
        foreach (var property in element.EnumerateObject())
        {
            if (property.Value.ValueKind == JsonValueKind.Number &&
                !string.Equals(property.Name, "num_classes", StringComparison.OrdinalIgnoreCase))
            {
                return property.Name;
            }
        }

        // Last resort: return the first property name only when it does not look like raw model metadata.
        var fallbackName = element.EnumerateObject().FirstOrDefault().Name ?? string.Empty;
        return IsRawModelMetadataKey(fallbackName) ? string.Empty : fallbackName;
    }

    private static bool IsRawModelMetadataKey(string key)
        => string.Equals(key, "probabilities", StringComparison.OrdinalIgnoreCase)
           || string.Equals(key, "scores", StringComparison.OrdinalIgnoreCase)
           || string.Equals(key, "top5", StringComparison.OrdinalIgnoreCase)
           || string.Equals(key, "num_classes", StringComparison.OrdinalIgnoreCase)
           || string.Equals(key, "classifier", StringComparison.OrdinalIgnoreCase);

    private static string ExtractTopLabelFromScores(JsonElement scores)
    {
        string topLabel = string.Empty;
        double topScore = double.MinValue;
        var sawNumeric = false;

        foreach (var property in scores.EnumerateObject())
        {
            if (TryReadDouble(property.Value, out var score) && score > topScore)
            {
                sawNumeric = true;
                topScore = score;
                topLabel = property.Name;
            }
        }

        return sawNumeric ? topLabel : string.Empty;
    }

    private static string ExtractTop5Label(JsonElement top5)
    {
        if (top5.ValueKind != JsonValueKind.Array)
        {
            return string.Empty;
        }

        using var enumerator = top5.EnumerateArray();
        if (!enumerator.MoveNext())
        {
            return string.Empty;
        }

        var first = enumerator.Current;
        if (first.ValueKind == JsonValueKind.Array)
        {
            using var pair = first.EnumerateArray();
            if (!pair.MoveNext())
            {
                return string.Empty;
            }

            return pair.Current.ValueKind == JsonValueKind.String
                ? pair.Current.GetString() ?? string.Empty
                : string.Empty;
        }

        return first.ValueKind == JsonValueKind.String
            ? first.GetString() ?? string.Empty
            : first.ToString();
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

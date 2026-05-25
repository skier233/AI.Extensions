using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using Pgvector;

namespace AI.Visual;

/// <summary>
/// CPU-side text encoder backed by an ONNX export.
/// Provides a fallback for visual semantic search when <c>nsfw_ai_server</c> is offline.
///
/// The encoder looks for the following files relative to the extension assembly directory:
/// <list type="bullet">
///   <item><c>text-encoder/text_encoder.onnx</c> — ONNX export of the text encoder</item>
///   <item><c>text-encoder/tokenizer.json</c> — tokenizer configuration</item>
/// </list>
/// Run <c>scripts/export-text-encoder.py</c> once to generate these files,
/// then place them in <c>extensions/AI.Visual/dist/text-encoder/</c> so they are copied
/// alongside the extension DLL at build time.
/// </summary>
internal sealed class AiVisualLocalTextEncoder : IDisposable
{
    private const string ModelSubdir = "text-encoder";
    private const string OnnxFileName = "text_encoder.onnx";
    private const string TokenizerFileName = "tokenizer.json";
    private const string StartupProbeText = "portrait";

    private readonly ILogger<AiVisualLocalTextEncoder>? _logger;
    private readonly AiVisualClipTokenizer? _tokenizer;
    private readonly InferenceSession? _session;
    private bool _disposed;

    /// <summary>Whether the ONNX model and tokenizer files were found and loaded.</summary>
    public bool IsAvailable => _session is not null && _tokenizer is not null;

    public AiVisualLocalTextEncoder(ILogger<AiVisualLocalTextEncoder>? logger = null)
    {
        _logger = logger;

        var assemblyDir = Path.GetDirectoryName(typeof(AiVisualLocalTextEncoder).Assembly.Location);
        if (assemblyDir is null)
        {
            _logger?.LogDebug("AiVisualLocalTextEncoder: could not determine assembly directory; local encoder will be unavailable.");
            return;
        }

        var modelDirCandidates = new[]
        {
            Path.Combine(assemblyDir, ModelSubdir),
            Path.Combine(assemblyDir, "dist", ModelSubdir),
        };

        var modelDir = modelDirCandidates.FirstOrDefault(Directory.Exists);
        if (modelDir is null)
        {
            _logger?.LogDebug(
                "AiVisualLocalTextEncoder: model files not found in {ModelDir}; local CPU encoder is unavailable. " +
                "Run scripts/export-text-encoder.py to generate them.",
                string.Join(", ", modelDirCandidates));
            return;
        }

        var onnxPath = Path.Combine(modelDir, OnnxFileName);
        var tokenizerPath = Path.Combine(modelDir, TokenizerFileName);

        if (!File.Exists(onnxPath) || !File.Exists(tokenizerPath))
        {
            _logger?.LogDebug(
                "AiVisualLocalTextEncoder: required files missing in {ModelDir}; local CPU encoder is unavailable.",
                modelDir);
            return;
        }

        AiVisualClipTokenizer? tokenizer = null;
        InferenceSession? session = null;
        try
        {
            tokenizer = new AiVisualClipTokenizer(tokenizerPath);

            var sessionOptions = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            };
            session = new InferenceSession(onnxPath, sessionOptions);

            var probe = RunInference(session, tokenizer, StartupProbeText);
            if (probe is null || probe.Length == 0)
            {
                throw new InvalidOperationException("Startup probe returned no embedding values.");
            }

            _tokenizer = tokenizer;
            _session = session;
            _logger?.LogInformation(
                "AiVisualLocalTextEncoder: loaded local CPU text encoder from {ModelDir}.",
                modelDir);
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(
                ex,
                "AiVisualLocalTextEncoder: model at {ModelDir} failed load or startup probe; local encoder is unavailable.",
                modelDir);
            session?.Dispose();
            tokenizer = null;
        }
    }

    /// <summary>
    /// Encodes <paramref name="text"/> to an L2-normalised 1024-dimensional vector.
    /// Returns <c>null</c> if the encoder is not available or inference fails.
    /// </summary>
    public Vector? TryEncode(string text)
    {
        if (_session is null || _tokenizer is null)
            return null;

        try
        {
            var embedding = RunInference(_session, _tokenizer, text);
            return embedding is null ? null : new Vector(embedding);
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "AiVisualLocalTextEncoder: inference failed for text '{Text}'.", text);
            return null;
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _session?.Dispose();
    }

    private static float[]? TryReadTensorValues(DisposableNamedOnnxValue output)
    {
        try
        {
            return output.AsTensor<float>().ToArray();
        }
        catch (InvalidCastException)
        {
            try
            {
                return output.AsTensor<Half>().Select(static value => (float)value).ToArray();
            }
            catch (InvalidCastException)
            {
                return null;
            }
        }
    }

    private static float[]? RunInference(InferenceSession session, AiVisualClipTokenizer tokenizer, string text)
    {
        var tokenIds = tokenizer.Encode(text);

        // Attention mask: 1 for real tokens, 0 for padding.
        var attentionMask = new long[AiVisualClipTokenizer.MaxLength];
        for (var i = 0; i < tokenIds.Length; i++)
            attentionMask[i] = tokenIds[i] != 1L ? 1L : 0L;

        var idsDimensions = new[] { 1, AiVisualClipTokenizer.MaxLength };
        var inputIds = tokenIds.Length == AiVisualClipTokenizer.MaxLength
            ? tokenIds
            : tokenIds.Take(AiVisualClipTokenizer.MaxLength).Concat(Enumerable.Repeat(1L, Math.Max(0, AiVisualClipTokenizer.MaxLength - tokenIds.Length))).ToArray();
        var attentionMaskValues = attentionMask.Length == AiVisualClipTokenizer.MaxLength
            ? attentionMask
            : attentionMask.Take(AiVisualClipTokenizer.MaxLength).Concat(Enumerable.Repeat(0L, Math.Max(0, AiVisualClipTokenizer.MaxLength - attentionMask.Length))).ToArray();

        var inputIdsTensor = new DenseTensor<long>(inputIds, idsDimensions);
        var attentionMaskTensor = new DenseTensor<long>(attentionMaskValues, idsDimensions);

        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
        };

        using var outputs = session.Run(inputs);
        var firstOutput = outputs.First();
        var raw = firstOutput.ValueType switch
        {
            OnnxValueType.ONNX_TYPE_TENSOR => TryReadTensorValues(firstOutput),
            _ => null,
        };

        if (raw is null || raw.Length == 0)
        {
            return null;
        }

        // L2-normalise (the export script should already normalise, but guard here too).
        var norm = MathF.Sqrt(raw.Sum(v => v * v));
        if (norm <= 1e-8f)
        {
            return null;
        }

        for (var i = 0; i < raw.Length; i++)
        {
            raw[i] /= norm;
        }

        if (raw.Any(static value => float.IsNaN(value) || float.IsInfinity(value)))
        {
            return null;
        }

        return raw;
    }

}

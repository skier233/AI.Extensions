using Tokenizers.HuggingFace.Tokenizer;

namespace AI.Visual;

internal sealed class AiVisualClipTokenizer : IDisposable
{
    public const int MaxLength = 77;

    private readonly Tokenizer _tokenizer;

    public AiVisualClipTokenizer(string tokenizerJsonPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerJsonPath);
        if (!File.Exists(tokenizerJsonPath))
        {
            throw new FileNotFoundException("Tokenizer file was not found.", tokenizerJsonPath);
        }

        _tokenizer = Tokenizer.FromFile(tokenizerJsonPath);
    }

    public long[] Encode(string text)
    {
        var encoding = _tokenizer.Encode(text ?? string.Empty, true).First();
        var ids = encoding.Ids.Select(static id => (long)id).ToArray();
        if (ids.Length == MaxLength)
            return ids;

        var output = new long[MaxLength];
        Array.Fill(output, 1L);
        Array.Copy(ids, output, Math.Min(ids.Length, MaxLength));
        return output;
    }

    public void Dispose()
    {
        _tokenizer.Dispose();
    }
}

namespace AI.Core;

internal static class AiPathMapper
{
    public static string NormalizePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return path;
        }

        var normalized = path.Trim().Replace('\\', '/');
        if (normalized.Length >= 2 && char.IsLetter(normalized[0]) && normalized[1] == ':')
        {
            if (normalized.Length == 2)
            {
                return normalized + "/";
            }

            if (normalized[2] != '/')
            {
                return normalized[..2] + "/" + normalized[2..];
            }
        }

        return normalized;
    }

    public static string MapPath(IReadOnlyList<AiPathMapping> mappings, string path)
    {
        var normalizedPath = NormalizePath(path);
        if (string.IsNullOrWhiteSpace(normalizedPath) || mappings.Count == 0)
        {
            return normalizedPath;
        }

        foreach (var mapping in mappings.OrderByDescending(static mapping => mapping.FromPrefix.Length))
        {
            if (!normalizedPath.StartsWith(mapping.FromPrefix, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var suffix = normalizedPath[mapping.FromPrefix.Length..].TrimStart('\\', '/');
            if (string.IsNullOrWhiteSpace(suffix))
            {
                return mapping.ToPrefix;
            }

            var separator = mapping.ToPrefix.Contains('\\', StringComparison.Ordinal) ? '\\' : '/';
            return $"{mapping.ToPrefix}{separator}{suffix.Replace('\\', separator).Replace('/', separator)}";
        }

        return normalizedPath;
    }

    public static IReadOnlyList<string> MapPaths(IReadOnlyList<AiPathMapping> mappings, IReadOnlyList<string> paths)
        => paths.Select(path => MapPath(mappings, path)).ToArray();
}
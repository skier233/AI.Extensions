using System.Numerics;
using System.Runtime.InteropServices;

namespace AI.Faces;

// SIMD dot product and norms for face identity matching (CosineSimilarity).
//
// Same arithmetic as the scalar loop it replaces: each element product is formed in single precision
// (exactly what `float * float` yields in C#), then widened to double and accumulated in double. Only the
// order of the double additions differs (lane-wise instead of strictly sequential), so results agree with
// the scalar version to within a few ULPs of a double.
internal static class FaceVectorMath
{
    internal static bool TryGetSpan(IReadOnlyList<float> values, out ReadOnlySpan<float> span)
    {
        switch (values)
        {
            case float[] array:
                span = array;
                return true;
            case List<float> list:
                span = CollectionsMarshal.AsSpan(list);
                return true;
            default:
                span = default;
                return false;
        }
    }

    // Returns dot, |left|^2 and |right|^2; the caller applies the same zero-norm rule as before.
    internal static (double Dot, double LeftNorm, double RightNorm) DotAndNorms(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        var length = left.Length;
        var dot = Vector<double>.Zero;
        var leftNorm = Vector<double>.Zero;
        var rightNorm = Vector<double>.Zero;
        var index = 0;

        if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
        {
            var width = Vector<float>.Count;
            var last = length - width;
            for (; index <= last; index += width)
            {
                var l = new Vector<float>(left.Slice(index, width));
                var r = new Vector<float>(right.Slice(index, width));

                Vector.Widen(l * r, out var productLow, out var productHigh);
                dot += productLow + productHigh;

                Vector.Widen(l * l, out var leftLow, out var leftHigh);
                leftNorm += leftLow + leftHigh;

                Vector.Widen(r * r, out var rightLow, out var rightHigh);
                rightNorm += rightLow + rightHigh;
            }
        }

        var dotSum = Vector.Sum(dot);
        var leftSum = Vector.Sum(leftNorm);
        var rightSum = Vector.Sum(rightNorm);

        for (; index < length; index++)
        {
            var l = left[index];
            var r = right[index];
            dotSum += l * r;
            leftSum += l * l;
            rightSum += r * r;
        }

        return (dotSum, leftSum, rightSum);
    }
}

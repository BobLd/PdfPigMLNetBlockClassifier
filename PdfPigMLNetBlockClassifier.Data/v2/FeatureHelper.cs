using System;
using System.Collections.Generic;
using System.Linq;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.DocumentLayoutAnalysis;
using UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor;
using static UglyToad.PdfPig.Core.PdfPath;
using UglyToad.PdfPig.Geometry;
using UglyToad.PdfPig.Outline;
using System.Text;

namespace PdfPigMLNetBlockClassifier.Data.v2
{
    public static class FeatureHelper
    {
        public static readonly string Header = "blockAspectRatio,charsCount,wordsCount,linesCount,pctNumericChars,pctAlphabeticalChars,pctSymbolicChars,pctBulletChars,deltaToHeight,pathsCount,pctBezierPaths,pctHorPaths,pctVertPaths,pctOblPaths,imagesCount,imageAvgProportion,bestNormEditDistance,label";

        public static readonly Dictionary<int, string> Categories = new Dictionary<int, string>()
        {
            { 0, "text" },
            { 1, "title" },
            { 2, "list" },
            { 3, "table" },
            { 4, "image" },
        };

        private static readonly char[] Bullets = new char[]
        {
            '•', 'o', '▪', '❖', '➢', '►', '✓', '➔', '⇨', '➪',
            '➨', '➫', '➬', '➭', '➮', '➯', '➱', '➲', '\u2023',
            '\u2043', '\u204C', '\u204D'
        };

        public static float[] GetFeatures(TextBlock textBlock, IEnumerable<PdfPath> paths, IEnumerable<IPdfImage> images,
            double averagePageFontHeight, double bboxArea, List<DocumentBookmarkNode> pageBookmarksNodes)
        {
            // text block features
            float blockAspectRatio = float.NaN;

            // Letters features
            float charsCount = 0;
            float pctNumericChars = 0;
            float pctAlphabeticalChars = 0;
            float pctSymbolicChars = 0;
            float pctBulletChars = 0;
            float deltaToHeight = float.NaN;   // might be problematic

            float wordsCount = 0;
            float linesCount = 0;
            float bestNormEditDistance = float.NaN;

            if (textBlock?.TextLines != null && textBlock.TextLines.Any())
            {
                blockAspectRatio = (float)Math.Round(textBlock.BoundingBox.Width / textBlock.BoundingBox.Height, 5);

                var avgHeight = averagePageFontHeight;

                var textLines = textBlock.TextLines;
                var words = textLines.SelectMany(tl => tl.Words).ToList();
                var letters = words.SelectMany(w => w.Letters).ToList();
                char[] chars = letters.SelectMany(l => l.Value).ToArray();

                charsCount = chars.Length;
                pctNumericChars = (float)Math.Round(chars.Count(c => char.IsNumber(c)) / charsCount, 5);
                pctAlphabeticalChars = (float)Math.Round(chars.Count(c => char.IsLetter(c)) / charsCount, 5);
                pctSymbolicChars = (float)Math.Round(chars.Count(c => !char.IsLetterOrDigit(c)) / charsCount, 5);
                pctBulletChars = (float)Math.Round(chars.Count(c => Bullets.Any(bullet => bullet == c)) / charsCount, 5);
                if (avgHeight != 0)
                {
                    deltaToHeight = (float)Math.Round(letters.Select(l => l.GlyphRectangle.Height).Average() / avgHeight, 5);
                }
             
                wordsCount = words.Count();
                linesCount = textLines.Count();

                if (pageBookmarksNodes != null)
                {
                    // http://www.unicode.org/reports/tr15/
                    var textBlockNormalised = textBlock.Text.Normalize(NormalizationForm.FormKC).ToLower();
                    foreach (var bookmark in pageBookmarksNodes)
                    {
                        // need to normalise both text
                        var bookmarkTextNormalised = bookmark.Title.Normalize(NormalizationForm.FormKC).ToLower();
                        var currentDist = Distances.MinimumEditDistanceNormalised(textBlockNormalised, bookmarkTextNormalised);
                        if (float.IsNaN(bestNormEditDistance) || currentDist < bestNormEditDistance)
                        {
                            bestNormEditDistance = (float)Math.Round(currentDist, 5);
                        }
                    }
                }
            }

            // Paths features
            float pathsCount = 0;
            float pctBezierPaths = 0;
            float pctHorPaths = 0;
            float pctVertPaths = 0;
            float pctOblPaths = 0;

            if (paths != null && paths.Count() > 0)
            {
                foreach (var path in paths)
                {
                    foreach (var command in path.Commands)
                    {
                        if (command is BezierCurve bezierCurve)
                        {
                            pathsCount++;
                            pctBezierPaths++;
                        }
                        else if (command is Line line)
                        {
                            pathsCount++;
                            if (line.From.X == line.To.X)
                            {
                                pctVertPaths++;
                            }
                            else if (line.From.Y == line.To.Y)
                            {
                                pctHorPaths++;
                            }
                            else
                            {
                                pctOblPaths++;
                            }
                        }
                    }
                }

                pctBezierPaths = (float)Math.Round(pctBezierPaths / pathsCount, 5);
                pctHorPaths = (float)Math.Round(pctHorPaths / pathsCount, 5);
                pctVertPaths = (float)Math.Round(pctVertPaths / pathsCount, 5);
                pctOblPaths = (float)Math.Round(pctOblPaths / pathsCount, 5);
            }

            // Images features
            float imagesCount = 0;
            float imageAvgProportion = 0;

            if (images != null && images.Count() > 0)
            {
                imagesCount = images.Count();
                imageAvgProportion = (float)(images.Average(i => i.Bounds.Area) / bboxArea);
            }

            return new float[]
            {
                blockAspectRatio, charsCount, wordsCount, linesCount, pctNumericChars,
                pctAlphabeticalChars, pctSymbolicChars, pctBulletChars, deltaToHeight,
                pathsCount, pctBezierPaths, pctHorPaths, pctVertPaths, pctOblPaths,
                imagesCount, imageAvgProportion, bestNormEditDistance
            };
        }

        public static IEnumerable<Letter> GetLettersInside(PdfRectangle bound, IEnumerable<Letter> letters)
        {
            return letters.Where(l => bound.IntersectsWith(l.GlyphRectangle));
        }

        static NearestNeighbourWordExtractor nearestNeighbourWordExtractor = NearestNeighbourWordExtractor.Instance;

        public static IReadOnlyList<Word> GetWords(IReadOnlyList<Letter> letters)
        {
            return nearestNeighbourWordExtractor.GetWords(letters).ToList();
        }

        public static IReadOnlyList<TextLine> GetLines(IReadOnlyList<Word> words)
        {
            return words.GroupBy(x => x.BoundingBox.Bottom).OrderByDescending(x => x.Key)
                .Select(x => new TextLine(x.ToList())).ToArray();
        }
        
        public static IEnumerable<IPdfImage> GetImagesInside(PdfRectangle bound, IEnumerable<IPdfImage> images)
        {
            return images.Where(b => b.Bounds.Left >= bound.Left &&
                                     b.Bounds.Right <= bound.Right &&
                                     b.Bounds.Bottom >= bound.Bottom &&
                                     b.Bounds.Top <= bound.Top);
        }

        public static IEnumerable<PdfPath> GetPathsInside(PdfRectangle bound, IEnumerable<PdfPath> paths)
        {
            return paths.Where(b => b.GetBoundingRectangle().HasValue)
                        .Where(b => b.GetBoundingRectangle().Value.Left >= bound.Left &&
                                    b.GetBoundingRectangle().Value.Right <= bound.Right &&
                                    b.GetBoundingRectangle().Value.Bottom >= bound.Bottom &&
                                    b.GetBoundingRectangle().Value.Top <= bound.Top);
        }
    }
}

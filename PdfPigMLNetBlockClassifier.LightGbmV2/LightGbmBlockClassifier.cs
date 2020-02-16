using Microsoft.ML;
using PdfPigMLNetBlockClassifier.Data.v2;
using System.Collections.Generic;
using System.Linq;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.DocumentLayoutAnalysis;
using UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter;
using UglyToad.PdfPig.Outline;
using UglyToad.PdfPig.Util;

namespace PdfPigMLNetBlockClassifier.LightGbmV2
{
    public class LightGbmBlockClassifier
    {
        private MLContext mlContext;
        private ITransformer mlModel;
        private PredictionEngine<BlockFeatures, BlockCategory> predEngine;

        public DataViewSchema OutputSchema => predEngine?.OutputSchema;

        private static readonly int[] MLNetCategories = new int[] { 2, 0, 3, 4, 1 };

        public LightGbmBlockClassifier(string modelPath)
        {
            mlContext = new MLContext();
            mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            predEngine = mlContext.Model.CreatePredictionEngine<BlockFeatures, BlockCategory>(mlModel);
        }

        public (string Prediction, float Score) Classify(TextBlock textBlock, IEnumerable<PdfPath> paths, IEnumerable<IPdfImage> images,
            double averagePageFontHeight, List<DocumentBookmarkNode> pageBookmarksNodes)
        {
            double bboxArea = textBlock.BoundingBox.Area;

            var letters = textBlock.TextLines.SelectMany(li => li.Words).SelectMany(w => w.Letters);


            var features = FeatureHelper.GetFeatures(
                textBlock, paths,
                images, averagePageFontHeight,
                textBlock.BoundingBox.Area,
                pageBookmarksNodes);

            BlockFeatures blockFeatures = new BlockFeatures()
            {
                BlockAspectRatio = features[0],
                CharsCount = features[1],
                WordsCount = features[2],
                LinesCount = features[3],
                PctNumericChars = features[4],
                PctAlphabeticalChars = features[5],
                PctSymbolicChars = features[6],
                PctBulletChars = features[7],
                DeltaToHeight = features[8],
                PathsCount = features[9],
                PctBezierPaths = features[10],
                PctHorPaths = features[11],
                PctVertPaths = features[12],
                PctOblPaths = features[13],
                ImagesCount = features[14],
                ImageAvgProportion = features[15],
                BestNormEditDistance = features[16],
            };
            var result = predEngine.Predict(blockFeatures);

            return (FeatureHelper.Categories[(int)result.Prediction], result.Score.Max());
        }

        public IEnumerable<(string Prediction, float Score, TextBlock Block)> Classify(Page page, IWordExtractor wordExtractor,
            IPageSegmenter pageSegmenter, Bookmarks bookmarks = null)
        {

            List<DocumentBookmarkNode> bookmarksNodes = bookmarks?.GetNodes()
                .Where(b => b is DocumentBookmarkNode)
                .Select(b => b as DocumentBookmarkNode)
                .Cast<DocumentBookmarkNode>()
                .Where(b => b.PageNumber == page.Number).ToList();

            var avgPageFontHeight = page.Letters.Select(l => l.GlyphRectangle.Height).Average();

            var words = wordExtractor.GetWords(page.Letters);
            var blocks = pageSegmenter.GetBlocks(words);

            foreach (var block in blocks)
            {
                var letters = block.TextLines.SelectMany(li => li.Words).SelectMany(w => w.Letters);
                var paths = FeatureHelper.GetPathsInside(block.BoundingBox, page.ExperimentalAccess.Paths);
                var images = FeatureHelper.GetImagesInside(block.BoundingBox, page.GetImages());

                var features = FeatureHelper.GetFeatures(
                    block, paths,
                    images, avgPageFontHeight,
                    block.BoundingBox.Area,
                    bookmarksNodes);

                BlockFeatures blockFeatures = new BlockFeatures()
                {
                    BlockAspectRatio = features[0],
                    CharsCount = features[1],
                    WordsCount = features[2],
                    LinesCount = features[3],
                    PctNumericChars = features[4],
                    PctAlphabeticalChars = features[5],
                    PctSymbolicChars = features[6],
                    PctBulletChars = features[7],
                    DeltaToHeight = features[8],
                    PathsCount = features[9],
                    PctBezierPaths = features[10],
                    PctHorPaths = features[11],
                    PctVertPaths = features[12],
                    PctOblPaths = features[13],
                    ImagesCount = features[14],
                    ImageAvgProportion = features[15],
                    BestNormEditDistance = features[16],
                };

                var result = predEngine.Predict(blockFeatures);

                yield return (FeatureHelper.Categories[(int)result.Prediction], result.Score.Max(), block);
            }
        }
    }
}

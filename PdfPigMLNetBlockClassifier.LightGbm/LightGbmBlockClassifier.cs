using Microsoft.ML;
using PdfPigMLNetBlockClassifier.Data;
using System.Collections.Generic;
using System.Linq;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.DocumentLayoutAnalysis;
using UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter;
using UglyToad.PdfPig.Util;

namespace PdfPigMLNetBlockClassifier.LightGbm
{
    public class LightGbmBlockClassifier
    {
        private MLContext mlContext;
        private ITransformer mlModel;
        private PredictionEngine<BlockFeatures, BlockCategory> predEngine;

        public LightGbmBlockClassifier(string modelPath)
        {
            mlContext = new MLContext();
            mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            predEngine = mlContext.Model.CreatePredictionEngine<BlockFeatures, BlockCategory>(mlModel);
        }

        public IEnumerable<(string Prediction, float Score, TextBlock Block)> Classify(Page page, IWordExtractor wordExtractor, IPageSegmenter pageSegmenter)
        {
            var words = wordExtractor.GetWords(page.Letters);
            var blocks = pageSegmenter.GetBlocks(words);

            foreach (var block in blocks)
            {
                var letters = block.TextLines.SelectMany(li => li.Words).SelectMany(w => w.Letters);
                var paths = FeatureHelper.GetPathsInside(block.BoundingBox, page.ExperimentalAccess.Paths);
                var images = FeatureHelper.GetImagesInside(block.BoundingBox, page.GetImages());
                var features = FeatureHelper.GetFeatures(page, block.BoundingBox, letters, paths, images);

                BlockFeatures blockFeatures = new BlockFeatures()
                {
                    CharsCount = features[0],
                    PctNumericChars = features[1],
                    PctAlphabeticalChars = features[2],
                    PctSymbolicChars = features[3],
                    PctBulletChars = features[4],
                    DeltaToHeight = features[5],
                    PathsCount = features[6],
                    PctBezierPaths = features[7],
                    PctHorPaths = features[8],
                    PctVertPaths = features[9],
                    PctOblPaths = features[10],
                    ImagesCount = features[11],
                    ImageAvgProportion = features[12]
                };

                var result = predEngine.Predict(blockFeatures);

                yield return (FeatureHelper.Categories[(int)result.Prediction], result.Score.Max(), block);
            }
        }
    }
}

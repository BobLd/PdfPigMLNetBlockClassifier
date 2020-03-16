using Microsoft.ML.Data;
using PdfPigMLNetBlockClassifier.Data.v2;
using PdfPigMLNetBlockClassifier.LightGbmV2;
using System;
using System.Collections.Generic;
using System.Linq;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter;
using UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor;
using UglyToad.PdfPig.Outline;

namespace PdfPigMLNetBlockClassifier
{
    class Program
    {
        static readonly string TRAIN_RAW_DATA_FILEPATH = @"D:\Datasets\Document Layout Analysis\PubLayNet\extracted\train";
        static readonly string TEST_RAW_DATA_FILEPATH = @"D:\Datasets\Document Layout Analysis\PubLayNet\extracted\val";

        static readonly string TRAIN_DATA_FILENAME = "features_train_v2.csv";
        static readonly string TEST_DATA_FILENAME = "features_val_v2.csv";

        static readonly string MODEL_NAME = "modelV2.zip";

        static void Main(string[] args)
        {
            // 1. Convert pdf documents and their PAGE xml ground truth to csv files 
            //DataGenerator.GetCsv(TEST_RAW_DATA_FILEPATH, 0, TEST_DATA_FILENAME);        // testing
            //DataGenerator.GetCsv(TRAIN_RAW_DATA_FILEPATH, 0, TRAIN_DATA_FILENAME);      // training

            // 2. Create the model
            //LightGbmModelBuilder.TrainModel(DataGenerator.GetDataPath(TRAIN_DATA_FILENAME), MODEL_NAME);

            // 3. Evaluate the model
            //LightGbmModelBuilder.Evaluate(MODEL_NAME, DataGenerator.GetDataPath(TEST_DATA_FILENAME));

            // 4. Load the trained classifier
            LightGbmBlockClassifier lightGbmBlockClassifier = new LightGbmBlockClassifier(LightGbmModelBuilder.GetModelPath(MODEL_NAME));

            var test = lightGbmBlockClassifier.OutputSchema["label"].HasSlotNames();
            
            NearestNeighbourWordExtractor nearestNeighbourWordExtractor = new NearestNeighbourWordExtractor();
            RecursiveXYCut recursiveXYCut = new RecursiveXYCut();

            using (var document = PdfDocument.Open("sample.pdf"))
            {
                var hasBookmarks = document.TryGetBookmarks(out Bookmarks bookmarks);

                for (var i = 0; i < document.NumberOfPages; i++)
                {
                    var page = document.GetPage(i + 1);

                    List<DocumentBookmarkNode> bookmarksNodes = bookmarks?.GetNodes()
                        .Where(b => b is DocumentBookmarkNode)
                        .Select(b => b as DocumentBookmarkNode)
                        .Cast<DocumentBookmarkNode>()
                        .Where(b => b.PageNumber == page.Number).ToList();

                    var avgPageFontHeight = page.Letters.Select(l => l.GlyphRectangle.Height).Average();

                    var words = nearestNeighbourWordExtractor.GetWords(page.Letters);
                    var blocks = recursiveXYCut.GetBlocks(words, page.Width / 3.0);

                    foreach (var block in blocks)
                    {
                        var paths = FeatureHelper.GetPathsInside(block.BoundingBox, page.ExperimentalAccess.Paths);
                        var images = FeatureHelper.GetImagesInside(block.BoundingBox, page.GetImages());

                        var pred = lightGbmBlockClassifier.Classify(block, paths, images, avgPageFontHeight, bookmarksNodes);

                        Console.WriteLine();
                        Console.WriteLine(pred.Prediction + " [" + pred.Score.ToString("0.0%") + "]");
                        Console.WriteLine(block.Text.Normalize(normalizationForm: System.Text.NormalizationForm.FormKC)); // remove ligatures
                    }
                }
            }

            Console.ReadKey();
        }
    }
}

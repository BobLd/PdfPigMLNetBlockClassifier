using PdfPigMLNetBlockClassifier.Data;
using PdfPigMLNetBlockClassifier.LightGbm;
using System;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter;
using UglyToad.PdfPig.DocumentLayoutAnalysis.ReadingOrderDetector;
using UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor;

namespace PdfPigMLNetBlockClassifier
{
    class Program
    {
        static readonly string TRAIN_RAW_DATA_FILEPATH = @"D:\Datasets\Document Layout Analysis\PubLayNet\extracted\train";
        static readonly string TEST_RAW_DATA_FILEPATH = @"D:\Datasets\Document Layout Analysis\PubLayNet\extracted\val";

        static readonly string TRAIN_DATA_FILENAME = "features_train.csv";
        static readonly string TEST_DATA_FILENAME = "features_val.csv";

        static void Main(string[] args)
        {
            // 1. Convert pdf documents and their PAGE xml ground truth to csv files 
            DataGenerator.GetCsv(TEST_RAW_DATA_FILEPATH, 0, TEST_DATA_FILENAME);        // testing
            DataGenerator.GetCsv(TRAIN_RAW_DATA_FILEPATH, 0, TRAIN_DATA_FILENAME);      // training

            // 2. Create the model
            LightGbmModelBuilder.CreateModel(DataGenerator.GetDataPath(TRAIN_DATA_FILENAME), "model.zip");

            // 3. Evaluate the model
            LightGbmModelBuilder.Evaluate("model.zip", DataGenerator.GetDataPath(TEST_DATA_FILENAME));

            // 4. Load the trained classifier
            LightGbmBlockClassifier lightGbmBlockClassifier = new LightGbmBlockClassifier(LightGbmModelBuilder.GetModelPath("model.zip"));

            using (var document = PdfDocument.Open("sample.pdf"))
            {
                for (var i = 0; i < document.NumberOfPages; i++)
                {
                    var page = document.GetPage(i + 1);
                    var classifiedBlocks = lightGbmBlockClassifier.Classify(page,
                                                            NearestNeighbourWordExtractor.Instance,
                                                            DocstrumBoundingBoxes.Instance);
                    var unsupervisedReadingOrderDetector = new UnsupervisedReadingOrderDetector();
                    
                    foreach (var block in classifiedBlocks)
                    {
                        Console.WriteLine();
                        Console.WriteLine(block.Prediction + " [" + block.Score.ToString("0.0%") + "]");
                        Console.WriteLine(block.Block.Text);
                    }
                }
            }

            Console.ReadKey();
        }
    }
}

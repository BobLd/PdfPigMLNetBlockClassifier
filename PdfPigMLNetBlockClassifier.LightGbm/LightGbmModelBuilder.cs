using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace PdfPigMLNetBlockClassifier.LightGbm
{
    public static class LightGbmModelBuilder
    {
        public static readonly string OutputFolderPath = @"../../../model";

        private static MLContext mlContext = new MLContext(seed: 1);

        public static void CreateModel(string trainDataFilePath, string outputModelName)
        {
            string outputFullPath = Path.Combine(OutputFolderPath, Path.ChangeExtension(outputModelName, "zip"));

            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<BlockFeatures>(
                                            path: trainDataFilePath,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Build training pipeline
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);

            // Evaluate quality of Model
            CrossValidate(mlContext, trainingDataView, trainingPipeline);

            // Train Model
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            // Save model
            SaveModel(mlContext, mlModel, outputFullPath, trainingDataView.Schema);
        }

        public static void Evaluate(string modelName, string testDataFilePath)
        {
            string modelFullPath = GetModelPath(modelName);

            // Create new MLContext
            MLContext mlContext = new MLContext();

            // Load model & create prediction engine
            ITransformer mlModel = mlContext.Model.Load(modelFullPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<BlockFeatures, BlockCategory>(mlModel);

            // Load Data
            IDataView testingDataView = mlContext.Data.LoadFromTextFile<BlockFeatures>(
                                            path: testDataFilePath,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            IDataView transformedTestingDataView = mlModel.Transform(testingDataView);

            Evaluate(mlContext, transformedTestingDataView);

            // Permutation Feature Importance
            // https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.permutationfeatureimportanceextensions.permutationfeatureimportance?view=ml-dotnet#Microsoft_ML_PermutationFeatureImportanceExtensions_PermutationFeatureImportance__1_Microsoft_ML_MulticlassClassificationCatalog_Microsoft_ML_ISingleFeaturePredictionTransformer___0__Microsoft_ML_IDataView_System_String_System_Boolean_System_Nullable_System_Int32__System_Int32_

            Console.WriteLine("=============== Permutation Feature Importance ===============");
            Console.WriteLine(@"PFI works by taking a labeled dataset, choosing a feature, and permuting the values for that 
feature across all the examples, so that each example now has a random value for the feature 
and the original values for all other features. The evaluation metric (e.g. micro-accuracy) is 
then calculated for this modified dataset, and the change in the evaluation metric from the 
original dataset is computed. The larger the change in the evaluation metric, the more 
important the feature is to the model. PFI works by performing this permutation analysis 
across all the features of a model, one after another. ");

            // Get the column name of input features.
            string[] featureColumns = testingDataView.Schema.Select(column => column.Name)
                                                            .Where(columnName => columnName != "label").ToArray();

            var predictor = ((mlModel as TransformerChain<ITransformer>).LastTransformer as TransformerChain<ITransformer>)
                                    .First() as MulticlassPredictionTransformer<OneVersusAllModelParameters>;

            var pfi = mlContext.MulticlassClassification.PermutationFeatureImportance(
                                                    predictor,
                                                    transformedTestingDataView,
                                                    labelColumnName: "label",
                                                    permutationCount: 30);

            // Now let's look at which features are most important to the model
            // overall. Get the feature indices sorted by their impact on
            // microaccuracy.
            var sortedIndices = pfi.Select((metrics, index) => new { index, metrics.MicroAccuracy })
                                   .OrderByDescending(feature => Math.Abs(feature.MicroAccuracy.Mean))
                                   .Select(feature => feature.index);

            Console.WriteLine("Feature\tChange in MicroAccuracy\t95% Confidence in "
                + "the Mean Change in MicroAccuracy");

            var microAccuracy = pfi.Select(x => x.MicroAccuracy).ToArray();

            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}\t{1:G4}\t{2:G4}",
                                  featureColumns[i],
                                  microAccuracy[i].Mean,
                                  1.96 * microAccuracy[i].StandardError);
            }
        }

        public static string GetModelPath(string modelName)
        {
            return Path.Combine(OutputFolderPath, Path.ChangeExtension(modelName, "zip"));
        }

        private static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("label", "label")
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "charsCount", "pctNumericChars", "pctAlphabeticalChars",
                                                                                                   "pctSymbolicChars", "pctBulletChars", "deltaToHeight",
                                                                                                   "pathsCount", "pctBezierPaths", "pctHorPaths",
                                                                                                   "pctVertPaths", "pctOblPaths", "imagesCount",
                                                                                                   "imageAvgProportion" }));

            // Set the training algorithm 
            var trainer = mlContext.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options()
            {
                NumberOfIterations = 150,
                LearningRate = 0.1158737f,
                NumberOfLeaves = 39,
                MinimumExampleCountPerLeaf = 50,
                UseCategoricalSplit = true,
                HandleMissingValue = false,
                MinimumExampleCountPerGroup = 50,
                MaximumCategoricalSplitPointCount = 32,
                CategoricalSmoothing = 10,
                L2CategoricalRegularization = 1,
                UseSoftmax = false,
                Booster = new GradientBooster.Options()
                {
                    L2Regularization = 0,
                    L1Regularization = 0
                },
                LabelColumnName = "label",
                FeatureColumnName = "Features",
                //UnbalancedSets = true,              // added by BobLd
            }).Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }
        
        private static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of training process ===============");
            return model;
        }

        private static void CrossValidate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "label");
            PrintMulticlassClassificationFoldsAverageMetrics(crossValidationResults);
        }

        private static void Evaluate(MLContext mlContext, IDataView testingDataView)
        {
            Console.WriteLine("=============== Evaluating to get model's accuracy metrics ===============");
            var evaluationResults = mlContext.MulticlassClassification.Evaluate(testingDataView, labelColumnName: "label");
            PrintMulticlassClassificationMetrics(evaluationResults);
        }

        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, modelRelativePath);
            Console.WriteLine("The model is saved to {0}", modelRelativePath);
        }

        private static void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    MacroAccuracy = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    MicroAccuracy = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"    LogLoss for class {i} \t= {metrics.PerClassLogLoss[i]:0.####}, the closer to 0, the better");
            }

            Console.WriteLine("    " + metrics.ConfusionMatrix.GetFormattedConfusionTable());

            for (int i = 0; i < metrics.ConfusionMatrix.PerClassPrecision.Count; i++)
            {
                var precision = metrics.ConfusionMatrix.PerClassPrecision[i];
                var recall = metrics.ConfusionMatrix.PerClassRecall[i];
                var f1Score = 2 * (precision * recall) / (precision + recall);
                Console.WriteLine($"    F1 Score for class {i} \t= {f1Score:0.####}, a value between 0 and 1, the closer to 1, the better");
            }

            Console.WriteLine($"************************************************************");
        }

        private static void PrintMulticlassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");
        }

        private static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        private static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }
    }
}

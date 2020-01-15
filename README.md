# PdfPig ML.Net Block Classifier
Proof of concept of training a simple Region Classifier using [PdfPig](https://github.com/UglyToad/PdfPig) and [ML.NET](https://github.com/dotnet/machinelearning). 
The objective is to classify each text block in a pdf document page as either __title__, __text__, __list__, __table__ and __image__.

[AutoML](https://docs.microsoft.com/en-us/dotnet/machine-learning/automate-training-with-model-builder) model builder was used. The model was
trained on a subset of the [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet#getting-data) dataset. See their license [here](https://cdla.io/permissive-1-0/).

## Metrics for multi-class classification model
```
MicroAccuracy = 0.9369, a value between 0 and 1, the closer to 1, the better
MacroAccuracy = 0.7482, a value between 0 and 1, the closer to 1, the better
LogLoss = 0.2092, the closer to 0, the better

LogLoss for class 0 (title)        = 0.2156, the closer to 0, the better
LogLoss for class 1 (list)         = 3.1245, the closer to 0, the better
LogLoss for class 2 (table)        = 0.3060, the closer to 0, the better
LogLoss for class 3 (text)         = 0.1094, the closer to 0, the better
LogLoss for class 4 (image)        = 0.2472, the closer to 0, the better

F1 Score for class 0 (title)       = 0.9003, a value between 0 and 1, the closer to 1, the better
F1 Score for class 1 (list)        = 0.0213, a value between 0 and 1, the closer to 1, the better
F1 Score for class 2 (table)       = 0.9361, a value between 0 and 1, the closer to 1, the better
F1 Score for class 3 (text)        = 0.9593, a value between 0 and 1, the closer to 1, the better
F1 Score for class 4 (image)       = 0.9161, a value between 0 and 1, the closer to 1, the better
```

## Confusion table
```
          ||=======================================================
PREDICTED ||     0 |     1 |     2 |     3 |     4 | Total | Recall
TRUTH     ||=======================================================
(title) 0 || 1,765 |     0 |     0 |   145 |     2 | 1,912 | 0.9231
(list)  1 ||     1 |     3 |     2 |   273 |     0 | 279   | 0.0108
(table) 2 ||     0 |     0 |   623 |    63 |     5 | 691   | 0.9016
(text)  3 ||   242 |     0 |    13 | 8,709 |     1 | 8,965 | 0.9714
(image) 4 ||     1 |     0 |     2 |     2 |    71 | 76    | 0.9342
          ||=======================================================
Precision ||0.8785 |1.0000 |0.9734 |0.9475 |0.8987 |
```

## Permutation Feature Importance
PFI works by taking a labeled dataset, choosing a feature, and permuting the values for that
feature across all the examples, so that each example now has a random value for the feature
and the original values for all other features. The evaluation metric (e.g. micro-accuracy) is
then calculated for this modified dataset, and the change in the evaluation metric from the
original dataset is computed. The larger the change in the evaluation metric, the more
important the feature is to the model. PFI works by performing this permutation analysis
across all the features of a model, one after another. - [Source]( https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.permutationfeatureimportanceextensions.permutationfeatureimportance?view=ml-dotnet#Microsoft_ML_PermutationFeatureImportanceExtensions_PermutationFeatureImportance__1_Microsoft_ML_MulticlassClassificationCatalog_Microsoft_ML_ISingleFeaturePredictionTransformer___0__Microsoft_ML_IDataView_System_String_System_Boolean_System_Nullable_System_Int32__System_Int32_)

|Feature              |Description| Change in MicroAccuracy | 95% Confidence in the Mean Change in MicroAccuracy |
|--------------------:|:-----------:|:-----------------------:|:--------------------------------------------------:|
|**charsCount**           |Characters count|-0.2192 |0.0008443 |
|**pctNumericChars**      |% of numeric characters|-0.04996 |0.0004363 |
|**deltaToHeight**        |Average delta to average page glyph height|-0.04155 |0.000428|
|**pctBulletChars**       |% of bullet characters|-0.01571 |0.0004034|
|**pctAlphabeticalChars** |% of alphabetical characters|-0.012 |0.0003245 |
|**pctSymbolicChars**     |% of symbolic characters|-0.01187 |0.0004204 |
|**pathsCount**           |Paths count|-0.01089 |0.0002144 |
|**pctHorPaths**          |% of horizontal paths|-0.002695 |0.0001318 |
|**imageAvgProportion**   |Average area covered by images|-0.001895 |0.00003909 |
|**pctVertPaths**         |% of vertical paths|-0.001403 |0.0001069 |
|**pctOblPaths**          |% of oblique paths|-0.0005032 |0.00003612 |
|**pctBezierPaths**       |% of Bezier curve paths|-0.00007828 |0.00001563 |
|**imagesCount**          |Images count|-0.00002796 |0.00002768 |

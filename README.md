# PdfPig ML.Net Block Classifier v2
Proof of concept of training a simple Region Classifier using [PdfPig](https://github.com/UglyToad/PdfPig) and [ML.NET](https://github.com/dotnet/machinelearning). 
The objective is to classify each text block in a __pdf document__ page as either __title__, __text__, __list__, __table__ and __image__.

[AutoML](https://docs.microsoft.com/en-us/dotnet/machine-learning/automate-training-with-model-builder) model builder was used. The model was
trained on a subset of the [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet#getting-data) dataset. See their license [here](https://cdla.io/permissive-1-0/).

| Generation | MicroAccuracy | MacroAccuracy |
|-----------:|:-------------:|:-------------:|
| **v1**         | 0.937         | 0.748         |
| **v2**         | 0.952         | 0.801         |

For v1 model results, see [here](https://github.com/BobLd/PdfPigMLNetBlockClassifier/blob/master/PdfPigMLNetBlockClassifier.LightGbm/README.md).

# Results
Results are based on PubLayNet's validation dataset, where the page segmentation is known. For real life use, a page segmenter will be needed (see PdfPig's [PageSegmenters](https://github.com/UglyToad/PdfPig/tree/master/src/UglyToad.PdfPig.DocumentLayoutAnalysis/PageSegmenter)). The quality of the page segmentation  will impact the results.
## Metrics for multi-class classification model
```
MicroAccuracy = 0.9523, a value between 0 and 1, the closer to 1, the better
MacroAccuracy = 0.8009, a value between 0 and 1, the closer to 1, the better
LogLoss = 3.5333, the closer to 0, the better

LogLoss for class 0 (title)         = 4.6289, the closer to 0, the better
LogLoss for class 1 (image)         = 1.849, the closer to 0, the better
LogLoss for class 2 (text)          = 2.5834, the closer to 0, the better
LogLoss for class 3 (list)          = 28.8412, the closer to 0, the better
LogLoss for class 4 (table)         = 2.8326, the closer to 0, the better

F1 Score for class 0 (title)       = 0.9305, a value between 0 and 1, the closer to 1, the better
F1 Score for class 1 (image)       = 0.9412, a value between 0 and 1, the closer to 1, the better
F1 Score for class 2 (text)        = 0.9691, a value between 0 and 1, the closer to 1, the better
F1 Score for class 3 (list)        = 0.2963, a value between 0 and 1, the closer to 1, the better
F1 Score for class 4 (table)       = 0.9611, a value between 0 and 1, the closer to 1, the better
```

## Confusion table
```
          ||=======================================================
PREDICTED ||     0 |     1 |     2 |     3 |     4 | Total | Recall
TRUTH     ||=======================================================
(title) 0 || 1,848 |     0 |    89 |     1 |     0 | 1,938 | 0.9536
(image) 1 ||     0 |    72 |     3 |     0 |     1 | 76    | 0.9474
(text)  2 ||   185 |     1 | 8,837 |    16 |     9 | 9,048 | 0.9767
(list)  3 ||     1 |     0 |   225 |    52 |     2 | 280   | 0.1857
(table) 4 ||     0 |     4 |    35 |     2 |   654 | 695   | 0.9410
          ||=======================================================
Precision ||0.9086 |0.9351 |0.9617 |0.7324 |0.9820 |
```

## Permutation Feature Importance
PFI works by taking a labeled dataset, choosing a feature, and permuting the values for that
feature across all the examples, so that each example now has a random value for the feature
and the original values for all other features. The evaluation metric (e.g. micro-accuracy) is
then calculated for this modified dataset, and the change in the evaluation metric from the
original dataset is computed. The larger the change in the evaluation metric, the more
important the feature is to the model. PFI works by performing this permutation analysis
across all the features of a model, one after another. - [Source]( https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.permutationfeatureimportanceextensions.permutationfeatureimportance?view=ml-dotnet#Microsoft_ML_PermutationFeatureImportanceExtensions_PermutationFeatureImportance__1_Microsoft_ML_MulticlassClassificationCatalog_Microsoft_ML_ISingleFeaturePredictionTransformer___0__Microsoft_ML_IDataView_System_String_System_Boolean_System_Nullable_System_Int32__System_Int32_)

### Micro Accuracy

|Feature                  | Description | Change in MicroAccuracy | 95% Confidence in the Mean Change in MicroAccuracy |
|------------------------:|:-----------:|:-----------------------:|:--------------------------------------------------:|
|**wordsCount**| Words count |-0.1144 	|0.0008237|
|**charsCount**| Characters count |-0.09849|0.0009115|
|**linesCount**| Lines count |-0.04255  |0.0005726|
|**blockAspectRatio**|Ratio between the block's width and height|-0.04159|0.0005134|
|**bestNormEditDistance**|Minimum edit distance between bookmark title and block text|-0.04016    |    0.0006307|
|**deltaToHeight**|Average delta to average page glyph height|-0.03689   |     0.0003823|
|**pctNumericChars**|% of numeric characters|-0.03375   |     0.0005174|
|**pctSymbolicChars**|% of symbolic characters|-0.01035   |     0.000358|
|**pctAlphabeticalChars**|% of alphabetical characters|  -0.00859    |    0.0003075|
|**pctBulletChars**|% of bullet characters|-0.004071   |    0.0002063|
|**pathsCount**|Paths count|-0.003661   |    0.0001184|
|**pctHorPaths**|% of horizontal paths|-0.003431   |    0.0001309|
|**imageAvgProportion**|Average area covered by images|-0.001684     |  4.68E-05|
|**pctVertPaths**|% of vertical paths|-0.001448   |    8.139E-05|
|**pctOblPaths**|% of oblique paths|-0.0001883   |   1.734E-05|
|**imagesCount**|Images count|-9.692E-05   |   1.127E-05|
|**pctBezierPaths**|% of Bezier curve paths|6.212E-22   |    1.352E-05|

### Macro Accuracy 

|Feature                  | Description | Change in MacroAccuracy | 95% Confidence in the Mean Change in MacroAccuracy |
|------------------------:|:-----------:|:-----------------------:|:--------------------------------------------------:|
|**charsCount**|Characters count|      		-0.1339 |		0.002263|
|**linesCount**| Lines count |      		-0.08378   |     0.0009925|
|**deltaToHeight**|Average delta to average page glyph height|   		-0.06461    |    0.001223|
|**blockAspectRatio**|Ratio between the block's width and height|        -0.05896     |   0.001163|
|**bestNormEditDistance**|Minimum edit distance between bookmark title and block text|    -0.05039     |   0.001419|
|**pctNumericChars**|% of numeric characters| 		-0.0475 	|	0.001258|
|**pathsCount**|Paths count|      		-0.0252 	|	0.000614|
|**wordsCount**|Words count|      		-0.01651   |     0.001655|
|**pctSymbolicChars**|% of symbolic characters|        -0.01166      |  0.001505|
|**pctHorPaths**|% of horizontal paths|     		-0.00984 |       0.0003186|
|**pctVertPaths**|% of vertical paths|    		-0.008558  |     0.0002925|
|**pctAlphabeticalChars**|% of alphabetical characters|    -0.006882    |   0.0009913|
|**imageAvgProportion**|Average area covered by images|      -0.006148     |  0.0001146|
|**pctBulletChars**|% of bullet characters|  		-0.005319  |     0.0008375|
|**pctOblPaths**|% of oblique paths|     		-0.002747   |    6.657E-05|
|**imagesCount**|Images count|     		-0.00268    |    3.903E-05|
|**pctBezierPaths**|% of Bezier curve paths|  		-4.078E-05  |    5.263E-05|

# TO DO
## Features
- Add a [decoration](https://github.com/UglyToad/PdfPig/blob/master/src/UglyToad.PdfPig.DocumentLayoutAnalysis/DecorationTextBlockClassifier.cs) score/flag
- Add block's area ratio: the ratio between block area and the page area, [cf.](http://www.cs.rug.nl/~aiellom/publications/ijdar.pdf)
- Add block's font style: an enumerated type, with possible values: regular, bold, italic, underline, [cf.](http://www.cs.rug.nl/~aiellom/publications/ijdar.pdf)
- Add % sparse lines in a block, for better table recognition [cf.](https://clgiles.ist.psu.edu/pubs/CIKM2008-table-boundaries.pdf)
- Font color distance from most common color

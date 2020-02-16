using Microsoft.ML.Data;

namespace PdfPigMLNetBlockClassifier.LightGbmV2
{
    public class BlockFeatures
    {
        [ColumnName("blockAspectRatio"), LoadColumn(0)]
        public float BlockAspectRatio { get; set; }


        [ColumnName("charsCount"), LoadColumn(1)]
        public float CharsCount { get; set; }


        [ColumnName("wordsCount"), LoadColumn(2)]
        public float WordsCount { get; set; }


        [ColumnName("linesCount"), LoadColumn(3)]
        public float LinesCount { get; set; }


        [ColumnName("pctNumericChars"), LoadColumn(4)]
        public float PctNumericChars { get; set; }


        [ColumnName("pctAlphabeticalChars"), LoadColumn(5)]
        public float PctAlphabeticalChars { get; set; }


        [ColumnName("pctSymbolicChars"), LoadColumn(6)]
        public float PctSymbolicChars { get; set; }


        [ColumnName("pctBulletChars"), LoadColumn(7)]
        public float PctBulletChars { get; set; }


        [ColumnName("deltaToHeight"), LoadColumn(8)]
        public float DeltaToHeight { get; set; }


        [ColumnName("pathsCount"), LoadColumn(9)]
        public float PathsCount { get; set; }


        [ColumnName("pctBezierPaths"), LoadColumn(10)]
        public float PctBezierPaths { get; set; }


        [ColumnName("pctHorPaths"), LoadColumn(11)]
        public float PctHorPaths { get; set; }


        [ColumnName("pctVertPaths"), LoadColumn(12)]
        public float PctVertPaths { get; set; }


        [ColumnName("pctOblPaths"), LoadColumn(13)]
        public float PctOblPaths { get; set; }


        [ColumnName("imagesCount"), LoadColumn(14)]
        public float ImagesCount { get; set; }


        [ColumnName("imageAvgProportion"), LoadColumn(15)]
        public float ImageAvgProportion { get; set; }


        [ColumnName("bestNormEditDistance"), LoadColumn(16)]
        public float BestNormEditDistance { get; set; }


        [ColumnName("label"), LoadColumn(17)]
        public float Label { get; set; }


    }
}

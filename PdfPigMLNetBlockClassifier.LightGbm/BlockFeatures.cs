using Microsoft.ML.Data;

namespace PdfPigMLNetBlockClassifier.LightGbm
{
    public class BlockFeatures
    {
        [ColumnName("charsCount"), LoadColumn(0)]
        public float CharsCount { get; set; }


        [ColumnName("pctNumericChars"), LoadColumn(1)]
        public float PctNumericChars { get; set; }


        [ColumnName("pctAlphabeticalChars"), LoadColumn(2)]
        public float PctAlphabeticalChars { get; set; }


        [ColumnName("pctSymbolicChars"), LoadColumn(3)]
        public float PctSymbolicChars { get; set; }


        [ColumnName("pctBulletChars"), LoadColumn(4)]
        public float PctBulletChars { get; set; }


        [ColumnName("deltaToHeight"), LoadColumn(5)]
        public float DeltaToHeight { get; set; }


        [ColumnName("pathsCount"), LoadColumn(6)]
        public float PathsCount { get; set; }


        [ColumnName("pctBezierPaths"), LoadColumn(7)]
        public float PctBezierPaths { get; set; }


        [ColumnName("pctHorPaths"), LoadColumn(8)]
        public float PctHorPaths { get; set; }


        [ColumnName("pctVertPaths"), LoadColumn(9)]
        public float PctVertPaths { get; set; }


        [ColumnName("pctOblPaths"), LoadColumn(10)]
        public float PctOblPaths { get; set; }


        [ColumnName("imagesCount"), LoadColumn(11)]
        public float ImagesCount { get; set; }


        [ColumnName("imageAvgProportion"), LoadColumn(12)]
        public float ImageAvgProportion { get; set; }


        [ColumnName("label"), LoadColumn(13)]
        public float Label { get; set; }
    }
}

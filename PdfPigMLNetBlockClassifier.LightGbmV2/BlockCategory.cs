using Microsoft.ML.Data;
using System;

namespace PdfPigMLNetBlockClassifier.LightGbmV2
{
    public class BlockCategory
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public Single Prediction { get; set; }

        public float[] Score { get; set; }
    }
}

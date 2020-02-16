using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Serialization;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.DocumentLayoutAnalysis.Export.PAGE;
using static UglyToad.PdfPig.DocumentLayoutAnalysis.Export.PAGE.PageXmlDocument;

namespace PdfPigMLNetBlockClassifier.Data.v1
{
    public static class DataGenerator
    {
        public static readonly string OutputFolderPath = @"../../../data";
        private static readonly string header = "charsCount,pctNumericChars,pctAlphabeticalChars,pctSymbolicChars,pctBulletChars,deltaToHeight,pathsCount,pctBezierPaths,pctHorPaths,pctVertPaths,pctOblPaths,imagesCount,imageAvgProportion,label";

        /// <summary>
        /// Generate a csv file of features. You will need the pdf documents and the ground truths in PAGE xml format.
        /// </summary>
        /// <param name="dataFolder">The path to the data folder. It should contain both the pdf files 
        /// and their corresponding ground truth xml files.</param>
        /// <param name="numberOfPdfDocs">Number of documents to concider.</param>
        public static void GetCsv(string dataFolder, int numberOfPdfDocs, string outputFileName)
        {
            string outputFullPath = GetDataPath(outputFileName);
            string outputErrorFullPath = Path.Combine(OutputFolderPath, "invalide_pdfs_" + Path.ChangeExtension(outputFileName, "txt"));

            ConcurrentBag<string> invalidPdfs = new ConcurrentBag<string>();
            ConcurrentBag<string> data = new ConcurrentBag<string>();

            int done = 0;

            DirectoryInfo d = new DirectoryInfo(dataFolder);
            var pdfFileLinks = d.GetFiles("*.pdf", SearchOption.TopDirectoryOnly);
            var maxPageNumber = d.GetFiles("*.xml", SearchOption.TopDirectoryOnly).Select(f => ParseXmlFileName(f.Name)).Max() + 1;

            numberOfPdfDocs = Math.Min(pdfFileLinks.Count(), numberOfPdfDocs);
            numberOfPdfDocs = numberOfPdfDocs == 0 ? pdfFileLinks.Count() : numberOfPdfDocs;

            var indexesSelected = GenerateRandom(numberOfPdfDocs, 0, pdfFileLinks.Length);

            Parallel.ForEach(indexesSelected, index =>
            {
                var pdfFile = pdfFileLinks[index];
                string fileName = pdfFile.Name;
                string xmlFileNameTemplate = fileName.Replace(".pdf", "_");

                var pageXmlLinksCandidates = Enumerable.Range(0, maxPageNumber).Select(i =>
                        Path.Combine(dataFolder, fileName.Replace(".pdf", "_" + string.Format("{0:00000}", i) + ".xml"))).ToArray();
                var pageXmlLinks = pageXmlLinksCandidates.Where(l => File.Exists(l)).Select(l => new FileInfo(l)).ToArray();

                if (pageXmlLinks.Length == 0)
                {
                    Console.BackgroundColor = ConsoleColor.DarkRed;
                    Console.WriteLine("Error for document '" + fileName + "': No PageXml files found");
                    Console.ResetColor();
                    return;
                }

                try
                {
                    var pagesNumbers = pageXmlLinks.Select(l => ParseXmlFileName(l.Name)).ToList();
                    List<float[]> localFeatures = new List<float[]>();
                    List<int> localCategories = new List<int>();
                    bool isValidDocument = true;

                    using (var doc = PdfDocument.Open(pdfFile.FullName))
                    {
                        // Checks if this pdf document looks to be valid
                        if ((pagesNumbers.Max() + 1) > doc.NumberOfPages)
                        {
                            // ignore this document as page number is not correct
                            Console.BackgroundColor = ConsoleColor.Red;
                            Console.WriteLine("Error for document '" + fileName + "': Ignoring this document as page number is not correct");
                            Console.ResetColor();
                            isValidDocument = false;
                        }

                        foreach (var pageXmlLink in pageXmlLinks)
                        {
                            if (!isValidDocument) break;

                            int pageNo = ParseXmlFileName(pageXmlLink.Name);
                            var page = doc.GetPage(pageNo + 1);

                            if (page.Rotation.Value != 0)
                            {
                                Console.BackgroundColor = ConsoleColor.Yellow;
                                Console.ForegroundColor = ConsoleColor.Black;
                                Console.WriteLine("Error for document '" + fileName + "': Ignoring page " + (pageNo + 1) + " because it is rotated");
                                Console.ResetColor();
                                continue;
                            }

                            var pageXml = Deserialize(pageXmlLink.FullName);

                            var blocks = pageXml.Page.Items;

                            foreach (var block in blocks)
                            {
                                int category = -1;
                                PdfRectangle bbox = new PdfRectangle();

                                if (block is PageXmlTextRegion textBlock)
                                {
                                    bbox = ParsePageXmlCoord(textBlock.Coords.Points, page.Height);
                                    switch (textBlock.Type)
                                    {
                                        case PageXmlTextSimpleType.Paragraph:
                                            category = 0;
                                            break;
                                        case PageXmlTextSimpleType.Heading:
                                            category = 1;
                                            break;
                                        case PageXmlTextSimpleType.LisLabel:
                                            category = 2;
                                            break;
                                        default:
                                            throw new ArgumentException("Unknown category");
                                    }

                                    if (FeatureHelper.GetLettersInside(bbox, page.Letters).Count() == 0)
                                    {
                                        Console.BackgroundColor = ConsoleColor.Red;
                                        Console.ForegroundColor = ConsoleColor.Black;
                                        Console.WriteLine("Error for document '" + fileName + "': Ignoring this document as an empty paragraph was found");
                                        Console.ResetColor();
                                        isValidDocument = false;
                                        break;
                                    }
                                }
                                else if (block is PageXmlTableRegion tableBlock)
                                {
                                    bbox = ParsePageXmlCoord(tableBlock.Coords.Points, page.Height);
                                    category = 3;
                                }
                                else if (block is PageXmlImageRegion imageBlock)
                                {
                                    bbox = ParsePageXmlCoord(imageBlock.Coords.Points, page.Height);
                                    category = 4;
                                }
                                else
                                {
                                    throw new ArgumentException("Unknown region type");
                                }

                                var letters = FeatureHelper.GetLettersInside(bbox, page.Letters).ToList();
                                var paths = FeatureHelper.GetPathsInside(bbox, page.ExperimentalAccess.Paths).ToList();
                                var images = FeatureHelper.GetImagesInside(bbox, page.GetImages());
                                var f = FeatureHelper.GetFeatures(page, bbox, letters, paths, images);

                                if (category == -1)
                                {
                                    throw new ArgumentException("Unknown category number.");
                                }

                                if (f != null)
                                {
                                    localFeatures.Add(f);
                                    localCategories.Add(category);
                                }
                            }
                        }
                    }

                    if (isValidDocument)
                    {
                        if (localFeatures.Count != localCategories.Count)
                        {
                            throw new ArgumentException("features and categories don't have the same size");
                        }

                        foreach (var line in localFeatures.Zip(localCategories, (f, c) => string.Join(",", f) + "," + c))
                        {
                            data.Add(line);
                        }
                    }
                    else
                    {
                        invalidPdfs.Add(pdfFile.Name);
                    }
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Error for document '" + fileName + "': " + ex.Message);
                    Console.ResetColor();
                }
                Console.WriteLine(done++);
            });

            List<string> csv = new List<string>() { header };
            csv.AddRange(data);
            File.WriteAllLines(outputFullPath, csv);
            File.WriteAllLines(outputErrorFullPath, invalidPdfs);

            Console.WriteLine("Done. Csv file saved in " + outputFullPath);
        }

        public static string GetDataPath(string fileName)
        {
            return Path.Combine(OutputFolderPath, Path.ChangeExtension(fileName, "csv"));
        }

        private static PageXmlDocument Deserialize(string xmlPath)
        {
            XmlSerializer serializer = new XmlSerializer(typeof(PageXmlDocument));

            using (var reader = XmlReader.Create(xmlPath))
            {
                return (PageXmlDocument)serializer.Deserialize(reader);
            }
        }

        private static PdfRectangle ParsePageXmlCoord(string points, double height)
        {
            string[] pointsStr = points.Split(' ');

            List<PdfPoint> pdfPoints = new List<PdfPoint>();

            foreach (var p in pointsStr)
            {
                string[] coord = p.Split(',');
                pdfPoints.Add(new PdfPoint(double.Parse(coord[0]), height - double.Parse(coord[1])));
            }

            return new PdfRectangle(pdfPoints.Min(p => p.X), pdfPoints.Min(p => p.Y), pdfPoints.Max(p => p.X), pdfPoints.Max(p => p.Y));
        }

        private static int ParseXmlFileName(string xmlFileName)
        {
            string split = xmlFileName.Split('_')[1].Replace(".xml", "");
            if (int.TryParse(split, out int pageNo))
            {
                return pageNo;
            }

            throw new ArgumentException("Cannot parse page number");
        }

        /// <summary>
        /// https://codereview.stackexchange.com/questions/61338/generate-random-numbers-without-repetitions 
        /// </summary>
        private static List<int> GenerateRandom(int count, int min, int max)
        {
            Random random = new Random(42);

            //  initialize set S to empty
            //  for J := N-M + 1 to N do
            //    T := RandInt(1, J)
            //    if T is not in S then
            //      insert T in S
            //    else
            //      insert J in S
            //
            // adapted for C# which does not have an inclusive Next(..)
            // and to make it from configurable range not just 1.

            if (max <= min || count < 0 ||
                    // max - min > 0 required to avoid overflow
                    (count > max - min && max - min > 0))
            {
                // need to use 64-bit to support big ranges (negative min, positive max)
                throw new ArgumentOutOfRangeException("Range " + min + " to " + max +
                        " (" + ((Int64)max - (Int64)min) + " values), or count " + count + " is illegal");
            }

            // generate count random values.
            HashSet<int> candidates = new HashSet<int>();

            // start count values before max, and end at max
            for (int top = max - count; top < max; top++)
            {
                // May strike a duplicate.
                // Need to add +1 to make inclusive generator
                // +1 is safe even for MaxVal max value because top < max
                if (!candidates.Add(random.Next(min, top + 1)))
                {
                    // collision, add inclusive max.
                    // which could not possibly have been added before.
                    candidates.Add(top);
                }
            }

            // load them in to a list, to sort
            List<int> result = candidates.ToList();

            // shuffle the results because HashSet has messed
            // with the order, and the algorithm does not produce
            // random-ordered results (e.g. max-1 will never be the first value)
            for (int i = result.Count - 1; i > 0; i--)
            {
                int k = random.Next(i + 1);
                int tmp = result[k];
                result[k] = result[i];
                result[i] = tmp;
            }
            return result;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using BitMiracle.LibTiff;
using BitMiracle.LibTiff.Classic;
using System.Drawing.Imaging;
using System.Windows.Media.Imaging;
using System.IO;
using System.Windows.Interop;
using System.Windows;
using System.Data.Objects;
using System.Runtime.InteropServices;
using Ionic.Zip;
using AForge;
using AForge.Imaging;
using AForge.Math;
using AForge.Math.Geometry;
using AForge.Imaging.Filters;

namespace CellMax_Circle_Detection
{
    public struct position
    {        
        public int i;
        public int j;
        public position(int i,int j)
        {
            this.i = i;
            this.j = j;
        }
    }
    class appendix
    {
        public void CircleDetection(string rootDir, string saveDir)
        {
            #region parameter setting

            string saveResultDir = "";
            string zipSaveDir = saveDir;

            List<string> subject_R_Dir = new List<string>();
            List<string> subject_G_Dir = new List<string>();
            List<string> subject_B_Dir = new List<string>();
            List<string> fileName_R = new List<string>();
            List<string> fileName_B = new List<string>();
            List<string> fileName_G = new List<string>();
            List<string> subjectName = new List<string>();

            float resizeRatio = 8;
            loadTIFFimage tif = new loadTIFFimage();
            List<CircleF> circle_BIG = new List<CircleF>();
            string[] BatchRawFile = new string[3];
            string[] BatchSaveFile = new string[4];

            string[] tmp = rootDir.Split('\\');
            for (int i = 0; i < tmp.Length - 1; i++)
            {
                saveResultDir = saveResultDir + tmp[i] + "/";
            }
            saveResultDir = saveResultDir + "tmp";

            #endregion

            folderString(rootDir, ref subject_R_Dir, ref subject_G_Dir, ref subject_B_Dir, ref subjectName, ref fileName_R, ref fileName_G, ref fileName_B);
            checkSaveDir(saveResultDir, subjectName);

            // ----- 10%

            int totalSubject = subject_R_Dir.Count;
            int width, height;
            CircleF circle = new CircleF();
            string newFile;

            #region Main Process
            for (int index = 0; index < totalSubject; index++)
            {
                width = 0;
                height = 0;

                BatchRawFile[0] = subject_R_Dir[index];
                BatchRawFile[1] = subject_G_Dir[index];
                BatchRawFile[2] = subject_B_Dir[index];

                BatchSaveFile[0] = saveResultDir + "/" + subjectName[index] + "/" + fileName_R[index];
                BatchSaveFile[1] = saveResultDir + "/" + subjectName[index] + "/" + fileName_G[index];
                BatchSaveFile[2] = saveResultDir + "/" + subjectName[index] + "/" + fileName_B[index];

                estimateCircle(subject_R_Dir[index], resizeRatio, ref circle, ref width, ref height, saveResultDir);  // estimate the circle in downsampling dimenstion

                // -------------- 

                circle_save_batch(BatchRawFile, BatchSaveFile, resizeRatio, circle);     // 2017.05.19
                // --------------

                BatchSaveFile[3] = saveResultDir + "/" + subjectName[index] + "/" + "Circle Information" + ".txt";

                StreamWriter writer = new StreamWriter(@BatchSaveFile[3]);
                writer.WriteLine("CenterX: {0}", (circle.Center.X * resizeRatio).ToString());
                writer.WriteLine("CenterY: {0}", (circle.Center.Y * resizeRatio).ToString());
                writer.WriteLine("Radius: {0}", (circle.Radius * resizeRatio).ToString());
                writer.Close();
                newFile = saveDir+"\\" + subjectName[index] + ".zip";
                compressZIP(BatchSaveFile, newFile);
                // -------------- 100%

                Array.Clear(BatchRawFile, 0, BatchRawFile.Length);
                Array.Clear(BatchSaveFile, 0, BatchSaveFile.Length);
                //Directory.Delete(subject_Dir[index], true);
            }
            #endregion

            Directory.Delete(saveResultDir, true);

        }

        private void folderString(string root, ref List<string> subject_R_Dir, ref List<string> subject_G_Dir, ref List<string> subject_B_Dir, ref List<string> subjectID, ref List<string> fileName_R, ref List<string> fileName_G, ref List<string> fileName_B)
        {
            string[] subject_Dir = Directory.GetDirectories(@root, "*", SearchOption.TopDirectoryOnly);
            string[] tmpDir;
            string[] tmpfile;
            int seperate = 0;

            foreach (string dir in subject_Dir)
            {
                subjectID.Add(dir.Remove(0, @root.Length + 1));
                tmpDir = Directory.GetFiles(@dir, "*.tif", SearchOption.TopDirectoryOnly);
                foreach (string file in tmpDir)
                {
                    string filetmp = "";
                    tmpfile = file.Split('_');
                    tmpfile[0] = tmpfile[0].Remove(0, dir.Length + 1);
                    seperate = tmpfile.Length;
                    for (int i = 0; i < seperate; i++)
                    {
                        if (i < seperate - 1)
                        {
                            filetmp = filetmp + tmpfile[i] + "_";
                        }
                        else
                            filetmp = filetmp + tmpfile[i];
                    }
                    if (string.Compare(tmpfile[0], "DAPI Selection", true) == 0)
                    {
                        fileName_B.Add(filetmp);
                        subject_B_Dir.Add(file);
                    }
                    if (string.Compare(tmpfile[0], "FITC Selection", true) == 0)
                    {
                        subject_G_Dir.Add(file);
                        fileName_G.Add(filetmp);
                    }
                    if (string.Compare(tmpfile[0], "TRITC-Rhoadmine Selection", true) == 0)
                    {
                        subject_R_Dir.Add(file);
                        fileName_R.Add(filetmp);
                    }
                }
            }
        }

        private void checkSaveDir(string saveTmpDir, List<string>subjectName)
        {
            string subjectFolder;
            if (Directory.Exists(saveTmpDir))
            {
                //...
            }
            else
            {
                //新增資料夾
                Directory.CreateDirectory(@saveTmpDir);
            }
            for (int i = 0; i < subjectName.Count; i++)
            {
                subjectFolder = saveTmpDir + "\\" + subjectName[i];
                if (Directory.Exists(subjectFolder))
                {
                    //...
                }
                else
                {
                    //新增資料夾
                    Directory.CreateDirectory(@subjectFolder);
                }
            }
        }

        #region remove the noise of canny edge image
        private int stdThreshold(Bitmap image)
        {
            int threshold = 0;
            short[][] imat = Img2mat(image);
            int maxValue = 0;
            double meanValue = 0;
            double std = 0;
            int width = image.Width;
            int height = image.Height;

            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (imat[i][j] > maxValue)
                        maxValue = imat[i][j];
                    meanValue = meanValue + imat[i][j];
                }
            }
            meanValue = meanValue / (width * height);
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    std = std + Math.Pow((imat[i][j] - meanValue), 2);
                }
            }
            std = Math.Sqrt(std / (width * height));
            threshold = (int)(meanValue + 5 * std);
            return threshold;
        }

        private Bitmap removeLowIntPixel(Bitmap image, int threshold)
        {
            int width = image.Width;
            int height = image.Height;
            Bitmap result = (Bitmap)image.Clone();
            short[][] imat = Img2mat(image);
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (imat[i][j] < threshold)
                    {
                        result.SetPixel(i, j, Color.Black);
                    }
                }
            }
            return result;
        }
        #endregion

        private Bitmap ResizeCircleFillProc(Bitmap image, float ratio)
        {
            loadTIFFimage tif = new loadTIFFimage();
            int width = image.Width;
            int height = image.Height;
            int wid = (int)(width / ratio);
            int hei = (int)(height / ratio);
            int threshold = 0;

            Bitmap resize = new Bitmap(image, wid, hei);

            Bitmap convert_image = tif.ConvertTo24bpp(resize);
            convert_image = CannyEdge(convert_image);
            threshold = stdThreshold(convert_image);
            CircleF circle = new CircleF();
            circleDetection4(ref convert_image, ref circle);
            float newCenterX = circle.Center.X * ratio;
            float newCenterY = circle.Center.Y * ratio;
            float newRadius = circle.Radius * ratio;
            convert_image = tif.ConvertTo24bpp(image);
            convert_image = CannyEdge(convert_image);
            Graphics g = Graphics.FromImage(convert_image);
            Pen godPen = new Pen(Color.Yellow, 10);
            g.DrawEllipse(godPen, newCenterX - newRadius, newCenterY - newRadius, newRadius * 2, newRadius * 2);
            //            g.DrawEllipse(godPen, circle.Center.X - circle.Radius, circle.Center.Y - circle.Radius, circle.Radius * 2, circle.Radius * 2);
            //            Image<Bgr, Byte> circleImage = new Image<Bgr, Byte>(convert_image);
            //            circleImage.Draw(circle, new Bgr(Color.YellowGreen), 2);
            //            convert_image = circleImage.ToBitmap();
            g.Dispose();
            return convert_image;
        }

        #region circle detection process
        private void estimateCircle(string fileName, float ratio, ref CircleF circle, ref int width, ref int height, string saveResultDir)
        {
            loadTIFFimage tif = new loadTIFFimage();
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Bitmap image = new Bitmap(fs);
            int w = (int)(image.Width / ratio);
            int h = (int)(image.Height / ratio);
            width = image.Width;
            height = image.Height;
            Bitmap resize = new Bitmap(image, w, h);

            Bitmap convert_image = tif.ConvertTo24bpp(resize);
            convert_image = CannyEdge(convert_image);
            circleDetection4(ref convert_image, ref circle);
            string[] partialName = fileName.Split('_');
            fs.Dispose();
        }
        #endregion

        #region save data

        public void circle_save_batch(string[] fileNames, string[] saveNames, float ratio, CircleF circle) // 2017.05.19
        {
            List<System.Drawing.Point> pointList = new List<System.Drawing.Point>();

            Tiff inputR = Tiff.Open(fileNames[0], "r");
            Tiff inputG = Tiff.Open(fileNames[1], "r");
            Tiff inputB = Tiff.Open(fileNames[2], "r");
            int width = inputR.GetField(TiffTag.IMAGEWIDTH)[0].ToInt();
            int height = inputR.GetField(TiffTag.IMAGELENGTH)[0].ToInt();
            int samplesPerPixel = inputR.GetField(TiffTag.SAMPLESPERPIXEL)[0].ToInt();
            int bitsPerSample = inputR.GetField(TiffTag.BITSPERSAMPLE)[0].ToInt();
            int photo = inputR.GetField(TiffTag.PHOTOMETRIC)[0].ToInt();

            int scanlineSize = inputR.ScanlineSize(); //width * 2
            byte[][] bufferR = new byte[height][];
            byte[][] bufferG = new byte[height][];
            byte[][] bufferB = new byte[height][];

            float centerX = circle.Center.X * ratio;
            float centerY = circle.Center.Y * ratio;
            float radius = circle.Radius * ratio;

            for (int i = 0; i < height; ++i)
            {
                bufferR[i] = new byte[scanlineSize];
                inputR.ReadScanline(bufferR[i], i);
                bufferG[i] = new byte[scanlineSize];
                inputG.ReadScanline(bufferG[i], i);
                bufferB[i] = new byte[scanlineSize];
                inputB.ReadScanline(bufferB[i], i);
            }
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    if (Math.Sqrt((i - centerX) * (i - centerX) + (j - centerY) * (j - centerY)) > radius)
                    {
                        bufferR[j][i * 2] = 0;
                        bufferR[j][i * 2 + 1] = 0;
                        bufferG[j][i * 2] = 0;
                        bufferG[j][i * 2 + 1] = 0;
                        bufferB[j][i * 2] = 0;
                        bufferB[j][i * 2 + 1] = 0;
                    }
                }
            }
            inputR.Dispose();
            inputB.Dispose();
            inputG.Dispose();

            Tiff outputR = Tiff.Open(saveNames[0], "w");
            Tiff outputG = Tiff.Open(saveNames[1], "w");
            Tiff outputB = Tiff.Open(saveNames[2], "w");

            Console.WriteLine("Save file: {0}", outputR);
            Console.WriteLine("Save file: {0}", outputG);
            Console.WriteLine("Save file: {0}", outputB);


            outputR.SetField(TiffTag.IMAGEWIDTH, width);
            outputR.SetField(TiffTag.IMAGELENGTH, height);
            outputR.SetField(TiffTag.SAMPLESPERPIXEL, samplesPerPixel);
            outputR.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample);
            outputR.SetField(TiffTag.ROWSPERSTRIP, outputR.DefaultStripSize(0));
            outputR.SetField(TiffTag.PHOTOMETRIC, photo);
            outputR.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
            //outputR.SetField(TiffTag.COMPRESSION, Compression.LZW);

            outputG.SetField(TiffTag.IMAGEWIDTH, width);
            outputG.SetField(TiffTag.IMAGELENGTH, height);
            outputG.SetField(TiffTag.SAMPLESPERPIXEL, samplesPerPixel);
            outputG.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample);
            outputG.SetField(TiffTag.ROWSPERSTRIP, outputG.DefaultStripSize(0));
            outputG.SetField(TiffTag.PHOTOMETRIC, photo);
            outputG.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
            //outputG.SetField(TiffTag.COMPRESSION, Compression.LZW);

            outputB.SetField(TiffTag.IMAGEWIDTH, width);
            outputB.SetField(TiffTag.IMAGELENGTH, height);
            outputB.SetField(TiffTag.SAMPLESPERPIXEL, samplesPerPixel);
            outputB.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample);
            outputB.SetField(TiffTag.ROWSPERSTRIP, outputB.DefaultStripSize(0));
            outputB.SetField(TiffTag.PHOTOMETRIC, photo);
            outputB.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
            //outputB.SetField(TiffTag.COMPRESSION, Compression.LZW);

            for (int i = 0; i < height; ++i)
            {
                outputR.WriteScanline(bufferR[i], i);
                outputG.WriteScanline(bufferG[i], i);
                outputB.WriteScanline(bufferB[i], i);
            }
            outputR.Close();
            outputG.Close();
            outputB.Close();


        }

        private void compressZIP(string[] fileNames, string saveName)
        {
            using (ZipFile zip = new ZipFile())
            {
                zip.CompressionLevel = Ionic.Zlib.CompressionLevel.Level2;
                zip.AddFiles(fileNames, "files");
                zip.Save(saveName);
            }
        } // 2017.05.19

        private void saveCircleInformation(string filename, CircleF circle)
        {
            StreamWriter sw = new StreamWriter(@"filename");
            sw.WriteLine("CenterX: {0}", circle.Center.X.ToString());
            sw.WriteLine("CenterY: {0}", circle.Center.Y.ToString());
            sw.WriteLine("Radius: {0}", circle.Radius.ToString());
            sw.Close();
        }

        #endregion

        #region image processing

        private short[][] Img2mat(Bitmap Img)
        {
            int i, j;
            short[][] imgaray = new short[Img.Width][];
            for (i = 0; i < Img.Width; i++)
            {
                imgaray[i] = new short[Img.Height];
            }

            for (i = 0; i < Img.Width; i++)
            {
                for (j = 0; j < Img.Height; j++)
                {
                    imgaray[i][j] = (short)((Img.GetPixel(i, j).R * 0.3) + (Img.GetPixel(i, j).G * 0.59) + (Img.GetPixel(i, j).B * 0.11));
                }
            }

            return imgaray;
        }

        private double[][] Img2mat_double(Bitmap Img)
        {
            int i, j;
            double[][] imgaray = new double[Img.Width][];
            for (i = 0; i < Img.Width; i++)
            {
                imgaray[i] = new double[Img.Height];
            }

            for (i = 0; i < Img.Width; i++)
            {
                for (j = 0; j < Img.Height; j++)
                {
                    imgaray[i][j] = (Img.GetPixel(i, j).R * 0.3) + (Img.GetPixel(i, j).G * 0.59) + (Img.GetPixel(i, j).B * 0.11);
                }
            }

            return imgaray;
        }

        // 8 bbp
        private Bitmap ColorToGrayscalev2(Bitmap bmp)
        {
            int w = bmp.Width,
            h = bmp.Height,
            r, ic, oc, bmpStride, outputStride, bytesPerPixel;
            PixelFormat pfIn = bmp.PixelFormat;
            ColorPalette palette;
            Bitmap output;
            BitmapData bmpData, outputData;

            //Create the new bitmap
            output = new Bitmap(w, h, PixelFormat.Format8bppIndexed);

            //Build a grayscale color Palette
            palette = output.Palette;
            for (int i = 0; i < 256; i++)
            {
                Color tmp = Color.FromArgb(255, i, i, i);
                palette.Entries[i] = Color.FromArgb(255, i, i, i);
            }
            output.Palette = palette;

            //No need to convert formats if already in 8 bit
            if (pfIn == PixelFormat.Format8bppIndexed)
            {
                output = (Bitmap)bmp.Clone();

                //Make sure the palette is a grayscale palette and not some other
                //8-bit indexed palette
                output.Palette = palette;

                return output;
            }
            //Get the number of bytes per pixel
            switch (pfIn)
            {
                case PixelFormat.Format24bppRgb: bytesPerPixel = 3; break;
                case PixelFormat.Format32bppArgb: bytesPerPixel = 4; break;
                case PixelFormat.Format32bppRgb: bytesPerPixel = 4; break;
                default: throw new InvalidOperationException("Image format not supported");
            }

            //Lock the images
            bmpData = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly,
            pfIn);
            outputData = output.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly,
            PixelFormat.Format8bppIndexed);
            bmpStride = bmpData.Stride;
            outputStride = outputData.Stride;

            //Traverse each pixel of the image
            unsafe
            {
                byte* bmpPtr = (byte*)bmpData.Scan0.ToPointer(),
                outputPtr = (byte*)outputData.Scan0.ToPointer();

                if (bytesPerPixel == 3)
                {
                    //Convert the pixel to it's luminance using the formula:
                    // L = .299*R + .587*G + .114*B
                    //Note that ic is the input column and oc is the output column
                    for (r = 0; r < h; r++)
                        for (ic = oc = 0; oc < w; ic += 3, ++oc)
                            outputPtr[r * outputStride + oc] = (byte)(int)
                            (0.299f * bmpPtr[r * bmpStride + ic] +
                            0.587f * bmpPtr[r * bmpStride + ic + 1] +
                            0.114f * bmpPtr[r * bmpStride + ic + 2]);
                }
                else //bytesPerPixel == 4
                {
                    //Convert the pixel to it's luminance using the formula:
                    // L = alpha * (.299*R + .587*G + .114*B)
                    //Note that ic is the input column and oc is the output column
                    for (r = 0; r < h; r++)
                        for (ic = oc = 0; oc < w; ic += 4, ++oc)
                            outputPtr[r * outputStride + oc] = (byte)(int)
                            ((bmpPtr[r * bmpStride + ic] / 255.0f) *
                            (0.299f * bmpPtr[r * bmpStride + ic + 1] +
                            0.587f * bmpPtr[r * bmpStride + ic + 2] +
                            0.114f * bmpPtr[r * bmpStride + ic + 3]));
                }
            }

            //Unlock the images
            bmp.UnlockBits(bmpData);
            output.UnlockBits(outputData);

            return output;
        }
        // 24 bbp
        private Bitmap GrayscaleToColor(Bitmap bmp)
        {
            Bitmap image;
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            image = (Bitmap)bmp.Clone(rect, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            return image;
        }

        private Bitmap CannyEdge(Bitmap img)
        {
            Bitmap output = ColorToGrayscalev2(img);
            CannyEdgeDetector filter = new CannyEdgeDetector();
            filter.GaussianSigma = 3;
            filter.GaussianSize = 19;
            filter.LowThreshold = 10;
            filter.HighThreshold = 20;
            filter.ApplyInPlace(output);
            output = GrayscaleToColor(output);
            //output = setEdgeThreshold(output, 0.33);
            return output;
        }



        #endregion

        #region Circle Detection

        private void circleDetection4(ref Bitmap img, ref CircleF trueCircle)
        {
            int cannyThreshold = 1;
            int accu_threshold = 30;
            int resolution = 1;
            int minDist = 35;
            int minRadius = (int)(img.Height * 0.9 * 0.5);
            int maxRadius = (int)(img.Width * 0.5);
            if (minRadius > maxRadius)
            {
                int tmp = minRadius;
                minRadius = maxRadius;
                maxRadius = tmp;
            }
            Bitmap cover_img = (Bitmap)img.Clone();
            int lowThreshold = stdThreshold(cover_img);
            cover_img = removeLowIntPixel(cover_img, lowThreshold);
            Bitmap bmp = ColorToGrayscalev2(cover_img);
            Image<Gray, Byte> image = new Image<Gray, Byte>(bmp);

            //reduce the image noise
            image._SmoothGaussian(3);
            CircleF[][] circles = image.HoughCircles(
                new Gray(cannyThreshold), //Canny algorithm high threshold
                //(the lower one will be twice smaller)  
                new Gray(accu_threshold), //accumulator threshold at the center detection stage
                resolution,             //accumulator resolution (size of the image / 2)
                minDist,            //minimum distance between two circles
                minRadius,            //min radius
                maxRadius);          //max radius

            // Hough Table
            int[] array = matchCircleTable(circles[0], img);
            int[] tmparray = (int[])array.Clone();
            Array.Sort(tmparray);
            int index = 0;
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] == tmparray[tmparray.Length - 1])
                {
                    index = i;
                    break;
                }
            }

            //Draw circles on image
            Image<Bgr, Byte> circleImage = new Image<Bgr, Byte>(img);
            trueCircle = circles[0][index];
            circleImage.Draw(trueCircle, new Bgr(Color.Tomato), 2);
            img = circleImage.ToBitmap();
            
            //img.Save(@"C:\Users\KR7110\Documents\TestData\testFile\Result\circleTemplate.jpg");
            //            Bitmap result = circleImage.ToBitmap();
            //            return result;
        }

        private int[] matchCircleTable(CircleF[] circles, Bitmap image)
        {
            // Compute the counting of each pair of radius and center from Hough Transform circle detection result.

            int[] array = new int[circles.Length];
            int neighbor = 3;
            short[][] imat = Img2mat(image);
            int width = image.Width;
            int height = image.Height;
            float x, y;
            float radius, centerX, centerY;
            short[,] neighborMat = new short[neighbor, neighbor];
            bool tag = false;
            int index = 0;

            foreach (CircleF circle in circles)
            {

                radius = circle.Radius;
                centerX = circle.Center.X;
                centerY = circle.Center.Y;
                for (int angle = 0; angle < 360; angle++)
                {
                    x = (float)(radius * Math.Cos(angle * Math.PI / 180F)) + centerX;
                    y = (float)(radius * Math.Sin(angle * Math.PI / 180F)) + centerY;
                    for (int i = -neighbor / 2; i <= neighbor / 2; i++)
                    {
                        for (int j = -neighbor / 2; j <= neighbor / 2; j++)
                        {
                            if ((x + i > 0 && x + i < width) && (y + j > 0 && y + j < height))
                            {
                                neighborMat[(int)(i + neighbor / 2), (int)(j + neighbor / 2)] = imat[(int)(x + i)][(int)(y + j)];
                            }
                        }
                    }
                    tag = neighborORnot(neighborMat, neighbor);
                    if (tag)
                    {
                        array[index] = array[index] + 1;
                    }
                    tag = false;
                }
                index += 1;
            }
            return array;
        }

        private bool neighborORnot(short[,] mat, int width)
        {
            // Return True if there exist at least one foreground pixel in the window.

            bool tag = false;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (i != width / 2 && j != width / 2)
                    {
                        if (mat[i, j] > 0)
                        {
                            tag = true;
                            break;
                        }
                    }
                }
                if (tag)
                {
                    break;
                }
            }
            return tag;
        }

        #endregion

    }
}

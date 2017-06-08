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
using OtsuThreshold;

namespace CTC_Preprocessing
{
    class appendix
    {
        public string saveFileName(string[] splitString)
        {
            string filename = "";
            for (int i = 0; i < splitString.Length; i++)
            {
                if (i < splitString.Length - 1)
                    filename = filename + splitString[i] + '_';
                else
                    filename = filename + splitString[i];

            }
            return filename;
        }

        public int stdThreshold(Bitmap image)
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
            threshold = (int)(meanValue + 6 * std);
            Console.WriteLine("maxValue: {0}, meanValue: {1}, std: {2}, threshold: {3}", maxValue, meanValue, std, threshold);
            return threshold;
        }

        public Bitmap removeLowIntPixel(Bitmap image, int threshold)
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

        public Bitmap ResizeCircleFillProc(Bitmap image, float ratio)
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
            circleDetection4(ref convert_image, ref circle, true);
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

        public void estimateCircle(string fileName, float ratio, ref CircleF [] circle, ref int width, ref int height, string saveResultDir)
        {
            loadTIFFimage tif = new loadTIFFimage();
            Otsu ot = new Otsu();
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Bitmap image = new Bitmap(fs);
            int w = (int)(image.Width / ratio);
            int h = (int)(image.Height / ratio);
            width = image.Width;
            height = image.Height;
            Console.WriteLine("Resizing into {0} times.", ratio);
            Bitmap resize = new Bitmap(image, w, h);
            Bitmap convert_image = tif.ConvertTo24bpp(resize);
            Console.WriteLine("Canny edge detection");
            convert_image = CannyEdge(convert_image);
            //circleDetection4(ref convert_image, ref circle);
            //circleDetection4(ref result, ref circle);
            doubleCircle(ref convert_image, ref circle);
            string[] partialName = fileName.Split('_');
            string saveDir;
            if (partialName.Length >= 2)
            {
                saveDir = saveResultDir + "/" + partialName[1] + ".jpg";
            }
            else
            {
                partialName = fileName.Split('\\');
                saveDir = saveResultDir + "/" + partialName[partialName.Length - 1] + ".jpg";
            }
            convert_image.Save(@saveDir);
            fs.Dispose();
        }

        public void circle_save_batch(string[] fileNames, string[] saveNames, float ratio, CircleF[] circle) // 2017.05.19
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

            float centerX1 = circle[0].Center.X * ratio;
            float centerY1 = circle[0].Center.Y * ratio;
            float radius1 = circle[0].Radius * ratio;
            float centerX2 = circle[1].Center.X * ratio;
            float centerY2 = circle[1].Center.Y * ratio;
            float radius2 = circle[1].Radius * ratio;

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
                    if (Math.Sqrt((i - centerX1) * (i - centerX1) + (j - centerY1) * (j - centerY1)) > radius1 || Math.Sqrt((i - centerX2) * (i - centerX2) + (j - centerY2) * (j - centerY2)) > radius2)
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

        public void compressZIP(string[] fileNames, string saveName)
        {
            using (ZipFile zip = new ZipFile())
            {
                zip.CompressionLevel = Ionic.Zlib.CompressionLevel.Level2;
                //zip.AddFiles(fileNames, "files");
                zip.AddFiles(fileNames,"");
                zip.Save(saveName);
            }
        }

        #region image processing

        public short[][] Img2mat(Bitmap Img)
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
        public Bitmap ColorToGrayscalev2(Bitmap bmp)
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
        public Bitmap GrayscaleToColor(Bitmap bmp)
        {
            Bitmap image;
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            image = (Bitmap)bmp.Clone(rect, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            return image;
        }

        public Bitmap CannyEdge(Bitmap img)
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

        public void circleDetection4(ref Bitmap img, ref CircleF trueCircle, bool first)
        {
            int cannyThreshold = 1;
            int accu_threshold = 30;
            int resolution = 1;
            int minDist = 35;
            int minRadius = (int)(img.Height * 0.8 * 0.5);
            int maxRadius = (int)(img.Width * 0.5);
            if (minRadius > maxRadius)
            {
                int tmp = minRadius;
                minRadius = maxRadius;
                maxRadius = tmp;
            }
            Bitmap cover_img = (Bitmap)img.Clone();
            //int lowThreshold = stdThreshold(cover_img);
            //cover_img = removeLowIntPixel(cover_img, lowThreshold);
            Console.WriteLine("max radius: {0}, min radius: {1}", maxRadius, minRadius);
            Bitmap bmp = ColorToGrayscalev2(cover_img);
            Image<Gray, Byte> image = new Image<Gray, Byte>(bmp);

            #region Hough Transform
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
            #endregion

            // Hough Table
            int[] array = matchCircleTable(circles[0], img);
            int[] tmparray = (int[])array.Clone();
            Array.Sort(tmparray);
            int index = -1;

            if (first)
            {
                for (int i = 0; i < array.Length; i++)
                {
                    if (array[i] == tmparray[tmparray.Length - 1] && index < 0)
                    {
                        index = i;
                        Console.WriteLine("Best index: {0}", index);
                        break;
                    }
                }
            }
            else
            {
                int shift = 1;
                bool jump = false;
                while (!jump)
                {
                    for (int i = 0; i < array.Length; i++)
                    {
                        if (array[i] == tmparray[tmparray.Length - shift] && index < 0)
                        {
                            if (circles[0][i].Radius == trueCircle.Radius && (circles[0][i].Center.X != trueCircle.Center.X || circles[0][i].Center.Y != trueCircle.Center.Y))
                            {
                                index = i;
                                Console.WriteLine("Best index: {0}", index);
                                jump = true;
                                break;
                            }
                            else if (Math.Floor(circles[0][i].Radius) != Math.Floor(trueCircle.Radius))
                            {
                                index = i;
                                Console.WriteLine("Best index: {0}", index);
                                jump = true;
                                break;
                            }

                        }
                    }
                    shift = shift + 1;
                    Console.WriteLine("Shift: {0}", shift);
                }
                
            }
            
            
            Console.WriteLine("*************************************************");
            Console.WriteLine("Circle Counting: {0}", circles[0].Length);
            Console.WriteLine("Circle 1: {0} / {1} / {2}", trueCircle.Center.X, trueCircle.Center.Y, trueCircle.Radius);
            Console.WriteLine("Circle 2: {0} / {1} / {2}", circles[0][index].Center.X, circles[0][index].Center.Y, circles[0][index].Radius);
            // -------------------------------------------------------------------------------
            //Draw circles on image
            Image<Bgr, Byte> circleImage = new Image<Bgr, Byte>(img);
            trueCircle = circles[0][index];
            circleImage.Draw(trueCircle, new Bgr(Color.Tomato), 2);
            //CircleF circle_2 = doubleCheckCircle(ref img, circles[0][index]);


            //img = circleImage.ToBitmap();

        }

        public int[] matchCircleTable(CircleF[] circles, Bitmap image)
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

        public bool neighborORnot(short[,] mat, int width)
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

        public void saveJPG(string fileName, string saveResultDir, float ratio, CircleF circle)
        {
            loadTIFFimage tif = new loadTIFFimage();
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Bitmap image = new Bitmap(fs);
            int w = (int)(image.Width / ratio);
            int h = (int)(image.Height / ratio);
            float centerX = circle.Center.X;
            float centerY = circle.Center.Y;
            float radius = circle.Radius;

            Console.WriteLine("Resizing into {0} times.", ratio);
            Bitmap resize = new Bitmap(image, w, h);

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    if (Math.Sqrt((i - centerX) * (i - centerX) + (j - centerY) * (j - centerY)) > radius)
                    {
                        resize.SetPixel(i, j, Color.Green);
                    }
                }
            }


            string[] partialName = fileName.Split('_');
            string saveDir = saveResultDir + partialName[1] + ".jpg";
            resize.Save(@saveDir);
            fs.Dispose();
        }

        public Bitmap binary_Component(Bitmap image)
        {
            Otsu ot = new Otsu();
            int threshold = ot.getOtsuThreshold(image);
            Bitmap result = ot.GetBinaryImg(image, threshold);
            // create filter
            BlobsFiltering filter = new BlobsFiltering();
            // configure filter
            filter.CoupledSizeFiltering = true;
            filter.MaxHeight = image.Height / 3;
            filter.MaxWidth = image.Width / 3;
            // apply the filter
            filter.ApplyInPlace(result);
            short[][] imat = Img2mat(result);
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    if(imat[i][j] == 0)
                    {
                        image.SetPixel(i, j, Color.Black);
                    }
                }
            }
            return image;
        }

        public Bitmap drawTest(Bitmap image, CircleF circle)
        {
            int width = image.Width;
            int height = image.Height;
            int centerX = (int)circle.Center.X;
            int centerY = (int)circle.Center.Y;
            float radius = circle.Radius;
            Bitmap result = (Bitmap)image.Clone();
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if ((Math.Sqrt(i - centerX) * (i - centerX) + (j - centerY) * (j - centerY)) > radius)
                    {
                        result.SetPixel(i, j, Color.Black);
                    }
                }
            }
            return result;
        }

        public CircleF doubleCheckCircle(ref Bitmap image, CircleF circle)
        {
            int width = image.Width;
            int height = image.Height;
            int centerX = (int)circle.Center.X;
            int centerY = (int)circle.Center.Y;
            float radius = circle.Radius;
            float r = 0;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    r = (float)Math.Sqrt((i - centerX) * (i - centerX) + (j - centerY) * (j - centerY));
                    if (r > radius)
                    {
                        image.SetPixel(i, j, Color.Black);
                    }
                }
            }
            int cannyThreshold = 1;
            int accu_threshold = 30;
            int resolution = 1;
            int minDist = 35;
            int minRadius = (int)(height * 0.8 * 0.5);
            int maxRadius = (int)(width * 0.5);
            if (minRadius > maxRadius)
            {
                int tmp = minRadius;
                minRadius = maxRadius;
                maxRadius = tmp;
            }
            Bitmap cover_img = (Bitmap)image.Clone();
            //int lowThreshold = stdThreshold(cover_img);
            //cover_img = removeLowIntPixel(cover_img, lowThreshold);
            Console.WriteLine("max radius: {0}, min radius: {1}", maxRadius, minRadius);
            Bitmap bmp = ColorToGrayscalev2(cover_img);
            Image<Gray, Byte> Timage = new Image<Gray, Byte>(bmp);

            //reduce the image noise
            Timage._SmoothGaussian(3);
            CircleF[][] circles = Timage.HoughCircles(
                new Gray(cannyThreshold), //Canny algorithm high threshold
                //(the lower one will be twice smaller)  
                new Gray(accu_threshold), //accumulator threshold at the center detection stage
                resolution,             //accumulator resolution (size of the image / 2)
                minDist,            //minimum distance between two circles
                minRadius,            //min radius
                maxRadius);          //max radius

            // Hough Table
            int[] array = matchCircleTable(circles[0], image);
            int[] tmparray = (int[])array.Clone();
            Array.Sort(tmparray);
            int index = -1;

            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] == tmparray[tmparray.Length - 1] && index < 0)
                {
                    index = i;
                    Console.WriteLine("Best index: {0}", index);
                    break;
                }
            }
            return circles[0][index];
        }

        public Bitmap preProcessingFilt(Bitmap image)
        {
            Bitmap result = (Bitmap)image.Clone();
            Otsu ot = new Otsu();
            int width = image.Width;
            int height = image.Height;
            int threshold = ot.getOtsuThreshold(result);
            result = ot.GetBinaryImg(result, threshold);
            // create filter
            BlobsFiltering filter = new BlobsFiltering();
            // configure filter
            filter.CoupledSizeFiltering = true;
            filter.MaxWidth = width / 3;
            filter.MaxHeight = height / 3;
            // apply the filter
            filter.ApplyInPlace(result);
            
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (result.GetPixel(i, j) == Color.Black)
                    {
                        image.SetPixel(i, j, Color.Black);
                    }
                }
            }

            return image;
        }

        public void doubleCircle(ref Bitmap img, ref CircleF[] circles)
        {
            Bitmap image_1 = (Bitmap)img.Clone();
            circleDetection4(ref image_1, ref circles[0], true);
            int width = img.Width;
            int height = img.Height;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (Math.Sqrt((i - circles[0].Center.X) * (i - circles[0].Center.X) + (j - circles[0].Center.Y) * (j - circles[0].Center.Y)) > circles[0].Radius)
                    {
                        image_1.SetPixel(i, j, Color.Black);
                        
                    }
                }
            }
            Image<Bgr, Byte> circleImage = new Image<Bgr, Byte>(image_1);
            circleImage.Draw(circles[0], new Bgr(Color.Tomato), 2);
            Bitmap image_2 = (Bitmap)image_1.Clone();
            CircleF tmp = new CircleF();
            tmp = circles[0];
            circleDetection4(ref image_2, ref tmp, false);
            circles[1] = tmp;
            circleImage.Draw(circles[1], new Bgr(Color.GreenYellow), 2);
            img = circleImage.ToBitmap();
            
        }

        #region sort radius and idx

        public int idealCircle(int[] array, CircleF[] circle, int bestIdx)
        {
            List<radiusAndSort> SortArray = new List<radiusAndSort>();
            
            int idx = 0;
            float[] ratio_array = new float[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                ratio_array[i] = (float)(array[i] / (2 * circle[i].Radius * Math.PI));
            }
            for (int i = 0; i < array.Length; i++)
            {
                SortArray.Add(new radiusAndSort { id = i.ToString(), radius = circle[i].Radius.ToString(), ratio = ratio_array[i].ToString() });
            }
            IEnumerable<radiusAndSort> query = from i in SortArray
                                               orderby i.ratio, i.id 
	                                 select i;
            foreach (radiusAndSort i in query)
            {
                Console.WriteLine(i.id + " " + i.radius + " " + i.ratio);
            } 
            //float[] ratio_sort_array = (float[])ratio_array.Clone();
            //Array.Sort(ratio_sort_array);
            //for (int i = 0; i < ratio_sort_array.Length; i++)
            //{
            //    if (ratio_sort_array[i] >= 0.7)
            //    {
            //        for (int j = 0; j < ratio_sort_array.Length; j++)
            //        {
            //        }
            //    }
            //}
            return idx;
        }
        
        public class radiusAndSort
        {
            public string id;
            public string radius;
            public string ratio;
        }

        #endregion

        #region Labeling

        public int[][] labelingImage(Bitmap image)
        {
            int width = image.Width;
            int height = image.Height;
            int[][] labelMat = new int[width][];
            short[][] imat = Img2mat(image);
            int time = 0;
            for (int i = 0; i < width; i++)
            {
                labelMat[i] = new int[height];
                for (int j = 0; j < height; j++)
                {
                    labelMat[i][j] = 0;
                    if (imat[i][j] == 255)
                    {
                        imat[i][j] = 1;
                    }
                    else
                    {
                        imat[i][j] = 0;
                    }
                }
            }

            int label_index = 0;
            bool check = false;

            while (!check)
            {
                check = true;
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        if (imat[i][j] == 1)
                        {
                            time = 0;
                            label_index = label_index + 1;
                            checkLabel(ref imat, ref labelMat, label_index, i, j, width, height, ref time);
                        }
                    }
                }
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        if (imat[i][j] == 1)
                        {
                            check = false;
                        }
                    }
                }
            }


            mergeLabel(ref labelMat, width, height);
            mergeLabel(ref labelMat, width, height);
            return labelMat;
        }

        private void checkLabel(ref short[][] mat, ref int[][] labelMat, int label_index, int x, int y, int width, int height, ref int time)
        {

            if ((x >= 0 && x < width) && (y >= 0 && y < height))
            {
                if (time < 3000)
                {
                    if (mat[x][y] == 1)
                    {
                        labelMat[x][y] = label_index;
                        mat[x][y] = 2;
                        time++;

                        checkLabel(ref mat, ref labelMat, label_index, x + 1, y, width, height, ref time);
                        checkLabel(ref mat, ref labelMat, label_index, x, y + 1, width, height, ref time);
                        checkLabel(ref mat, ref labelMat, label_index, x - 1, y, width, height, ref time);
                        checkLabel(ref mat, ref labelMat, label_index, x, y - 1, width, height, ref time);
                    }
                }
                else
                {
                    return;
                }
            }
        }

        private void mergeLabel(ref int[][] labelMat, int width, int height)
        {
            int minLabel = 0;
            for (int i = 1; i < width - 1; i++)
            {
                for (int j = 1; j < height - 1; j++)
                {
                    if (labelMat[i][j] != 0)
                    {
                        minLabel = (width * height);
                        int x = i - 1;
                        while (x <= i + 1)
                        {
                            int y = j - 1;
                            while (y <= j + 1)
                            {
                                if ((minLabel > labelMat[x][y]) && (labelMat[x][y] > 0))
                                {
                                    minLabel = labelMat[x][y];
                                }
                                y++;
                            }
                            x++;
                        }
                        x = i - 1;
                        while (x <= i + 1)
                        {
                            int y = j - 1;
                            while (y <= j + 1)
                            {
                                if (labelMat[x][y] != 0)
                                {
                                    labelMat[x][y] = minLabel;
                                }
                                y++;
                            }
                            x++;
                        }
                    }
                }
            }
        }

        public void labelCount(int[][] labelMat, ref int[] count, /*ref int big_index,*/int width, int height)
        {
            int maxValue = 0;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (maxValue < labelMat[i][j])
                    {
                        maxValue = labelMat[i][j];
                    }
                }
            }
            count = new int[maxValue + 1];
            for (int i = 0; i < maxValue + 1; i++)
            {
                count[i] = 0;
            }

            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (labelMat[i][j] > 0)
                    {
                        count[labelMat[i][j]]++;
                    }
                }
            }

        }

        public int[,] keepBiggestOne(int[,] input, int width, int height)
        {
            int[,] output = new int[width, height];

            int maxLabel = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (maxLabel < input[i, j])
                    {
                        maxLabel = input[i, j];
                    }
                }
            }
            maxLabel = maxLabel + 1;
            int[] count = new int[maxLabel];
            for (int i = 0; i < maxLabel; i++)
            {
                count[i] = 0;
            }
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (input[i, j] > 0)
                    {
                        count[input[i, j]]++;
                    }
                }
            }
            int index = 0;
            int maxCount = 0;
            for (int i = 0; i < maxLabel; i++)
            {
                if (maxCount < count[i])
                {
                    maxCount = count[i];
                    index = i;
                }
            }
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (input[j, i] == index)
                    {
                        output[i, j] = 255;
                    }
                }
            }
            return output;
        }

        public int[,] connectComponent(Bitmap image)
        {
            int w = image.Width;
            int h = image.Height;
            int label = 0;
            int[,] map = new int[h, w];
            Stack<int[]> s = new Stack<int[]>();

            /* construct the binary map */
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Color rgb = image.GetPixel(x, y);
                    int value = rgb.R;

                    if (value == 0) map[y, x] = rgb.R;
                    else map[y, x] = -1;
                }
            }

            /* find connected component */
            for (int y = 0; y < image.Height; y++) // traverse from top to buttom, left to right
            {
                for (int x = 0; x < image.Width; x++)
                {
                    if (map[y, x] == 0)
                    { // foreground,push then go through its neighbors
                        label++;
                        map[y, x] = label;
                        s.Push(new int[2] { y, x });
                        while (s.Count > 0)
                        {
                            int[] pos = s.Pop();

                            for (int i = -1; i <= 1; i++)
                            {
                                for (int j = -1; j <= 1; j++)
                                {
                                    int iy = pos[0] + i;
                                    int ix = pos[1] + j;
                                    if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                    {
                                        if (map[iy, ix] == 0)
                                        {
                                            map[iy, ix] = label;
                                            s.Push(new int[2] { iy, ix });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Console.WriteLine("Component Count: " + label.ToString());
            return map;

        }
        
        #endregion

        #region Seldom Use
        public void saveNewRawData(string fileName, string saveName, bool[][] map)
        {
            using (Tiff input = Tiff.Open(fileName, "r"))
            {
                int width = input.GetField(TiffTag.IMAGEWIDTH)[0].ToInt();
                int height = input.GetField(TiffTag.IMAGELENGTH)[0].ToInt();
                int samplesPerPixel = input.GetField(TiffTag.SAMPLESPERPIXEL)[0].ToInt();
                int bitsPerSample = input.GetField(TiffTag.BITSPERSAMPLE)[0].ToInt();
                int photo = input.GetField(TiffTag.PHOTOMETRIC)[0].ToInt();

                int scanlineSize = input.ScanlineSize(); //width * 2
                byte[][] buffer = new byte[height][];
                for (int i = 0; i < height; ++i)
                {
                    buffer[i] = new byte[scanlineSize];
                    input.ReadScanline(buffer[i], i);
                }

                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++)
                    {
                        if (!map[i][j])
                        {
                            buffer[j][i * 2] = 255;
                            buffer[j][i * 2 + 1] = 255;
                        }
                    }
                }


                using (Tiff output = Tiff.Open(saveName, "w"))
                {
                    output.SetField(TiffTag.IMAGEWIDTH, width);
                    output.SetField(TiffTag.IMAGELENGTH, height);
                    output.SetField(TiffTag.SAMPLESPERPIXEL, samplesPerPixel);
                    output.SetField(TiffTag.BITSPERSAMPLE, bitsPerSample);
                    output.SetField(TiffTag.ROWSPERSTRIP, output.DefaultStripSize(0));
                    output.SetField(TiffTag.PHOTOMETRIC, photo);
                    output.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                    output.SetField(TiffTag.COMPRESSION, Compression.LZW);

                    // change orientation of the image
                    //                    output.SetField(TiffTag.ORIENTATION, Orientation.RIGHTBOT);

                    for (int i = 0; i < height; ++i)
                    {
                        output.WriteScanline(buffer[i], i);
                    }

                }
            }
        }

        public void reCompress(string[] input)
        {
            ZipFile unzip;
            string tmp;
            string unZipPath;
            string ZipPath;
            int count = input.Length;
            string[] subFile;
            for (int i = 0; i < count; i++)
            {
                unzip = ZipFile.Read(input[i]);
                Console.WriteLine("{0} re-ZIP ING...", input[i]);
                unZipPath = input[i].Replace(".zip", "");
                foreach (ZipEntry e in unzip)
                {
                    e.Extract(unZipPath, ExtractExistingFileAction.OverwriteSilently);
                }
                tmp = unZipPath + '/' + "files/";
                subFile = Directory.GetFiles(tmp);
                ZipPath = unZipPath + ".zip";
                unzip.Dispose();
                compressZIP(subFile, ZipPath);
                Directory.Delete(unZipPath, true);
            }
            //ZipFile unzip = ZipFile.Read(input);
            //string unZipPath = input.Replace(".zip", "");
            //foreach (ZipEntry e in unzip)
            //{
            //    e.Extract(unZipPath, ExtractExistingFileAction.OverwriteSilently);
            //}
        }

        public void saveNewRawDataBatch(string[] fileNames, string[] saveNames, bool[][] map)
        {
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
                    if (!map[i][j])
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

        public void saveCircleInformation(string filename, CircleF circle)
        {
            StreamWriter sw = new StreamWriter(@"filename");
            sw.WriteLine("CenterX: {0}", circle.Center.X.ToString());
            sw.WriteLine("CenterY: {0}", circle.Center.Y.ToString());
            sw.WriteLine("Radius: {0}", circle.Radius.ToString());
            sw.Close();
            Console.WriteLine("circle information saved.");
        }

        public bool[][] generateCircleMap(int width, int height, float ratio, CircleF circle)
        {
            bool[][] map = new bool[width][];
            Console.WriteLine("map initializing...");
            for (int i = 0; i < width; i++)
            {
                map[i] = new bool[height];
                for (int j = 0; j < height; j++)
                {
                    map[i][j] = false;
                }
            }
            Bitmap image = new Bitmap(width, height);
            float centerX = circle.Center.X * ratio;
            float centerY = circle.Center.Y * ratio;
            float radius = circle.Radius * ratio;
            Graphics g = Graphics.FromImage(image);
            SolidBrush redBrush = new SolidBrush(Color.Red);
            g.FillEllipse(redBrush, centerX - radius, centerY - radius, radius * 2, radius * 2);
            Console.WriteLine("map generating...");
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (image.GetPixel(i, j).R == 255 && image.GetPixel(i, j).G == 0)
                        map[i][j] = true;
                    //if (imat[i][j] > 0)
                    //    map[i][j] = false;
                }
            }
            return map;
        }
        #endregion
    }
}

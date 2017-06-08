using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;
using AForge;
using AForge.Imaging;
using AForge.Math;
using AForge.Math.Geometry;
using AForge.Imaging.Filters;
using OtsuThreshold;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;

using System.Diagnostics;
using System.Runtime.InteropServices;


namespace CTC_detection_Test_v01
{
    class ImageProcess
    {

        #region Morphological operation
        #endregion

        public Bitmap erosion(Bitmap image)
        {
            // create filter
            Erosion filter = new Erosion();
            // convert 24 to 8 bpp
            Bitmap bmp = color2gray(image);
            // apply the filter
            filter.Apply(bmp);
            // convert 8 to 24 bpp
            image = gray2color(bmp);
            return image;
        }

        public Bitmap dilation(Bitmap image)
        {
            // create filter
            Dilatation filter = new Dilatation();
            // convert 24 to 8 bpp
            Bitmap bmp = color2gray(image);
            // apply the filter
            filter.Apply(bmp);
            // convert 8 to 24 bpp
            image = gray2color(bmp);
            return image;
        }

        public Bitmap opening(Bitmap image)
        {
            // create filter
            Opening filter = new Opening();
            // convert 24 to 8 bpp
            Bitmap bmp = color2gray(image);
            // apply the filter
            filter.Apply(bmp);
            // convert 8 to 24 bpp
            image = gray2color(bmp);
            return image;
        }

        public Bitmap closing(Bitmap image)
        {
            // create filter
            Closing filter = new Closing();
            // convert 24 to 8 bpp
            Bitmap bmp = color2gray(image);
            // apply the filter
            filter.Apply(bmp);
            // convert 8 to 24 bpp
            image = gray2color(bmp);
            return image;
        }

        public Bitmap bottomhat(Bitmap image)
        {
            // create filter
            BottomHat filter = new BottomHat();
            // convert 24 to 8 bpp
            Bitmap bmp = color2gray(image);
            // apply the filter
            filter.Apply(bmp);
            // convert 8 to 24 bpp
            image = gray2color(bmp);
            return image;
        }

        public Bitmap tophat(Bitmap image)
        {
            // create filter
            TopHat filter = new TopHat();
            // convert 24 to 8 bpp
            Bitmap bmp = color2gray(image);
            // apply the filter
            filter.Apply(bmp);
            // convert 8 to 24 bpp
            image = gray2color(bmp);
            return image;
        }

        public Bitmap hitandmiss(Bitmap image)
        {
            // define kernel to remove pixels on the right side of objects
            // (pixel is removed, if there is white pixel on the left and
            // black pixel on the right)
            short[,] se = new short[,] {
                                            { -1, -1, -1 },
                                            {  1,  1,  0 },
                                            { -1, -1, -1 }
                                        };
            // create filter
            HitAndMiss filter = new HitAndMiss(se, HitAndMiss.Modes.Thinning);
            // apply the filter
            filter.ApplyInPlace(image);
            return image;
        }

        private Otsu ot = new Otsu();

        #region image <-> matrix
        #endregion

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

        public Bitmap mat2Img(short[][] imat, int width, int height)
        {

            Bitmap output = new Bitmap(width, height);

            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    output.SetPixel(i, j, Color.FromArgb(imat[i][j], imat[i][j], imat[i][j]));
                }
            }
            return output;
        }

        public short[][] InitialMat(int width, int height)
        {
            short[][] output = new short[width][];
            for (int i = 0; i < width; i++)
            {
                output[i] = new short[height];
            }
            return output;
        }

        #region image convert
        #endregion

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

        public void inverseIntensity(ref List<Bitmap> img)
        {
            int width, height;
            for (int i = 0; i < img.Count; i++)
            {
                width = img[i].Width;
                height = img[i].Height;
                short[][] imat = Img2mat(img[i]);
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        imat[x][y] = (short)(255 - imat[x][y]);
                    }
                }
                img[i] = mat2Img(imat, width, height);
            }

        }

        public static Bitmap ColorToGrayscale(Bitmap bmp)
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

        public Bitmap inv_image(Bitmap image)
        {
            int width = image.Width;
            int height = image.Height;
            Bitmap output = new Bitmap(width, height);
            short[][] imat = Img2mat(image);
            int intensity = 0;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    intensity = 255 - imat[i][j];
                    output.SetPixel(i, j, Color.FromArgb(intensity, intensity, intensity));
                }
            }
            return output;
        }

        public Bitmap color2gray(Bitmap image)
        {
            // create grayscale filter (BT709)  -  8bpp
            Grayscale bfilter = new Grayscale(0.2125, 0.7154, 0.0721);
            // apply the filter
            Bitmap grayImage = bfilter.Apply(image);
            return grayImage;
        }

        public Bitmap gray2color(Bitmap grayImage)
        {
            // create filter    - 24bpp
            GrayscaleToRGB cfilter = new GrayscaleToRGB();
            // apply the filter
            Bitmap rgbImage = cfilter.Apply(grayImage);
            return rgbImage;
        }

        public Bitmap ConvertTo16bpp(Bitmap img)
        {
            var bmp = new Bitmap(img.Width, img.Height,
                          System.Drawing.Imaging.PixelFormat.Format16bppRgb555);
            //var bmp = new Bitmap(img.Width, img.Height,
            //              System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
            using (var gr = Graphics.FromImage(bmp))
                gr.DrawImage(img, new Rectangle(0, 0, img.Width, img.Height));
            return bmp;
        }

        #region enhancement
        #endregion

        public Bitmap histEq(Bitmap img)
        {
            int matw, math;
            int N;

            double[] array = new double[256];
            double[] temp = new double[256];
            short[] hist = new short[256];
            short[][] mat = Img2mat(img);
            matw = img.Width;
            math = img.Height;
            N = matw * math;
            Bitmap result = new Bitmap(matw, math);
            short[][] hmat = new short[matw][];
            for (int i = 0; i < matw; i++)
            {
                hmat[i] = new short[math];
            }
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = 0;
                hist[i] = 0;
                temp[i] = 0;
            }
            for (int i = 0; i < matw; i++)
            {
                for (int j = 0; j < math; j++)
                {
                    array[mat[i][j]] = array[mat[i][j]] + 1;
                }
            }
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = array[i] / N;
                for (int j = 0; j < i; j++)
                {
                    temp[i] = (temp[i] + array[j]);
                    //hist[i] = (short)Math.Round(((hist[i] + array[j]) * 255), 0);
                }
                hist[i] = (short)Math.Round((temp[i] * 255), 0);
            }
            for (int intensity = 0; intensity < hist.Length; intensity++)
            {
                for (int i = 0; i < matw; i++)
                {
                    for (int j = 0; j < math; j++)
                    {
                        if (mat[i][j] == intensity)
                        {
                            hmat[i][j] = hist[intensity];
                        }
                    }
                }
            }

            result = mat2Img(hmat, matw, math);
            return result;
        }

        public Bitmap SetBrightness(Bitmap img, int brightness)
        {
            Bitmap temp = (Bitmap)img;
            Bitmap bmap = (Bitmap)temp.Clone();
            if (brightness < -255) brightness = -255;
            if (brightness > 255) brightness = 255;
            Color c;
            for (int i = 0; i < bmap.Width; i++)
            {
                for (int j = 0; j < bmap.Height; j++)
                {
                    c = bmap.GetPixel(i, j);
                    int cR = c.R + brightness;
                    int cG = c.G + brightness;
                    int cB = c.B + brightness;

                    if (cR < 0) cR = 1;
                    if (cR > 255) cR = 255;

                    if (cG < 0) cG = 1;
                    if (cG > 255) cG = 255;

                    if (cB < 0) cB = 1;
                    if (cB > 255) cB = 255;

                    bmap.SetPixel(i, j, Color.FromArgb((byte)cR, (byte)cG, (byte)cB));
                }
            }
            return bmap;
        }

        public Bitmap SetContrast(Bitmap img, double contrast)
        {
            Bitmap temp = (Bitmap)img;
            Bitmap bmap = (Bitmap)temp.Clone();
            if (contrast < -100) contrast = -100;
            if (contrast > 100) contrast = 100;
            contrast = (100.0 + contrast) / 100.0;
            contrast *= contrast;
            Color c;
            for (int i = 0; i < bmap.Width; i++)
            {
                for (int j = 0; j < bmap.Height; j++)
                {
                    c = bmap.GetPixel(i, j);
                    double pR = c.R / 255.0;
                    pR -= 0.5;
                    pR *= contrast;
                    pR += 0.5;
                    pR *= 255;
                    if (pR < 0) pR = 0;
                    if (pR > 255) pR = 255;

                    double pG = c.G / 255.0;
                    pG -= 0.5;
                    pG *= contrast;
                    pG += 0.5;
                    pG *= 255;
                    if (pG < 0) pG = 0;
                    if (pG > 255) pG = 255;

                    double pB = c.B / 255.0;
                    pB -= 0.5;
                    pB *= contrast;
                    pB += 0.5;
                    pB *= 255;
                    if (pB < 0) pB = 0;
                    if (pB > 255) pB = 255;

                    bmap.SetPixel(i, j, Color.FromArgb((byte)pR, (byte)pG, (byte)pB));
                }
            }
            return bmap;
        }

        public Bitmap localContrast(Bitmap img)
        {
            short[][] imat = Img2mat(img);
            int w = img.Width;
            int h = img.Height;
            double[][] delta = new double[w][];
            double[][] ebar = new double[w][];
            double[][] c = new double[w][];
            double[][] newc = new double[w][];
            short[][] result = InitialMat(w, h);
            for (int i = 0; i < w; i++)
            {
                delta[i] = new double[h];
                ebar[i] = new double[h];
                c[i] = new double[h];
                newc[i] = new double[h];
            }
            for (int i = 1; i < w - 1; i++)
            {
                for (int j = 1; j < h - 1; j++)
                {
                    double average = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            if (m != n)
                            {
                                average = average + imat[i + m][j + n];
                            }
                        }
                    }
                    average = average / 8;
                    delta[i][j] = Math.Abs(imat[i][j] - average);
                }
            }
            for (int i = 1; i < w - 1; i++)
            {
                for (int j = 1; j < h - 1; j++)
                {
                    double tmp = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            ebar[i][j] = ebar[i][j] + (delta[i + m][j + n] * imat[i + m][j + n]);
                            tmp = tmp + delta[i + m][j + n];
                        }
                    }
                    ebar[i][j] = ebar[i][j] / tmp;
                    c[i][j] = Math.Abs(imat[i][j] - ebar[i][j]) / Math.Abs(imat[i][j] + ebar[i][j]);
                    newc[i][j] = Math.Sqrt(c[i][j]);
                    if (imat[i][j] <= ebar[i][j])
                    {
                        result[i][j] = (short)Math.Round(ebar[i][j] * (1 - newc[i][j]) / (1 + newc[i][j]));
                    }
                    else
                    {
                        result[i][j] = (short)Math.Round(ebar[i][j] * (1 + newc[i][j]) / (1 - newc[i][j]));
                    }
                    if (result[i][j] > 255)
                        result[i][j] = 255;
                    if (result[i][j] < 0)
                        result[i][j] = 0;
                }
            }
            Bitmap output = mat2Img(result, w, h);
            return output;
        }

        public Bitmap statisticContrast(Bitmap img)
        {
            short[][] imat = Img2mat(img);
            int w = img.Width;
            int h = img.Height;
            Bitmap I_hist = histEq(img);
            short[][] result = InitialMat(w, h);
            short[][] hmat = Img2mat(I_hist);
            int max = 0;
            int min = 100;
            double mean = 0;
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    if (imat[i][j] > max)
                        max = imat[i][j];
                    if (imat[i][j] < min)
                        min = imat[i][j];
                    mean = mean + imat[i][j];
                }
            }
            mean = mean / (w * h);
            double t = Math.Abs(((max - min) / 2) - mean);
            for (int i = 1; i < w - 1; i++)
            {
                for (int j = 1; j < h - 1; j++)
                {
                    double xmean = 0;
                    double std = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            xmean = xmean + imat[i + m][j + n];
                        }
                    }
                    xmean = xmean / 9;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            std = std + Math.Pow((imat[i + m][j + n] - xmean), 2);
                        }
                    }
                    std = Math.Sqrt(std / 9);
                    double diff = imat[i][j] - std;
                    if (diff > t)
                    {
                        result[i][j] = hmat[i][j];
                    }
                    else
                    {
                        result[i][j] = imat[i][j];
                    }
                }
            }
            Bitmap output = mat2Img(result, w, h);
            return output;

        }

        public Bitmap powerLaw(Bitmap image, double gamma)
        {
            int w = image.Width;
            int h = image.Height;
            int max = 0;
            int c = 1;
            short[][] imat = Img2mat(image);
            double[][] rmat = new double[w][];
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    if (max < imat[i][j])
                        max = imat[i][j];
                }
            }
            double dmax = 0;
            double dmin = 255;
            for (int i = 0; i < w; i++)
            {
                rmat[i] = new double[h];
                for (int j = 0; j < h; j++)
                {
                    rmat[i][j] = (double)imat[i][j] / max;
                    rmat[i][j] = c * Math.Pow(rmat[i][j], gamma);
                    if (rmat[i][j] > dmax)
                        dmax = rmat[i][j];
                    if (rmat[i][j] < dmin)
                        dmin = rmat[i][j];
                }
            }
            Bitmap result = new Bitmap(w, h);
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    rmat[i][j] = Math.Round(255 * (rmat[i][j] - dmin) / (dmax - dmin));
                    if (rmat[i][j] > 255)
                        rmat[i][j] = 255;
                    result.SetPixel(i, j, Color.FromArgb((int)rmat[i][j], (int)rmat[i][j], (int)rmat[i][j]));
                }
            }
            return result;




        }

        public Bitmap clahe(Bitmap image, double clipLimit, Size size)
        {
            Image<Bgr, Byte> tmp = new Image<Bgr, byte>(image);
            Image<Gray, Byte> gTmp = tmp.Convert<Gray, Byte>();

            CvInvoke.cvCLAHE(gTmp, clipLimit, size, gTmp);
            tmp = gTmp.Convert<Bgr, Byte>();
            Bitmap result = tmp.ToBitmap();
            return result;
        }

        public Bitmap anisotropicFilter(Bitmap image, double K)
        {
            int width = image.Width;
            int height = image.Height;
            Bitmap gimage = LapacianEdgeDetector(image);
            short[][] gmat = Img2mat(gimage);
            double normValue = norm(gmat, width, height);
            double g = 1 / (1 + Math.Pow(normValue / K, 2));
            int[][] tmp = new int[width][];
            for (int i = 0; i < width; i++)
            {
                tmp[i] = new int[height];
            }
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    tmp[i][j] = (int)(g * gmat[i][j]);
                }
            }
            short[][] div = divergence(tmp, width, height);
            short[][] rmat = InitialMat(width, height);
            short[][] imat = Img2mat(image);

            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    rmat[i][j] = (short)(imat[i][j] + div[i][j]);
                    if (rmat[i][j] > 255)
                    {
                        rmat[i][j] = 255;
                    }
                }
            }
            Bitmap result = mat2Img(rmat, width, height);
            return result;
        }

        public Bitmap anisodiff2D(Bitmap image, int num_iter)
        {
            double delta_t = (double)1 / 7;
            double kappa = 100;
            int option = 1;
            int width = image.Width;
            int height = image.Height;
            double[][] im = new double[width][];
            double[][] diff_im = new double[width][];
            short[][] imat = Img2mat(image);
            for (int i = 0; i < width; i++)
            {
                im[i] = new double[height];
                diff_im[i] = new double[height];
                for (int j = 0; j < height; j++)
                {
                    im[i][j] = (double)imat[i][j];
                    diff_im[i][j] = (double)imat[i][j];
                }
            }
            int dx = 1;
            int dy = 1;
            double dd = Math.Sqrt(2);
            double[][] cN, cS, cW, cE, cNE, cSE, cSW, cNW;

            #region mask initial
            short[][] hN = new short[3][];
            short[][] hS = new short[3][];
            short[][] hE = new short[3][];
            short[][] hW = new short[3][];
            short[][] hNE = new short[3][];
            short[][] hSE = new short[3][];
            short[][] hSW = new short[3][];
            short[][] hNW = new short[3][];

            double[][] nablaN; double[][] nablaS; double[][] nablaW; double[][] nablaE; double[][] nablaNE; double[][] nablaSE; double[][] nablaSW; double[][] nablaNW;
            for (int i = 0; i < 3; i++)
            {
                hN[i] = new short[3];
                hS[i] = new short[3];
                hE[i] = new short[3];
                hW[i] = new short[3];
                hNE[i] = new short[3];
                hSE[i] = new short[3];
                hSW[i] = new short[3];
                hNW[i] = new short[3];

            }
            hN[0][0] = 0; hN[0][1] = 1; hN[0][2] = 0;
            hN[1][0] = 0; hN[1][1] = -1; hN[1][2] = 0;
            hN[2][0] = 0; hN[2][1] = 0; hN[2][2] = 0;

            hS[0][0] = 0; hS[0][1] = 0; hS[0][2] = 0;
            hS[1][0] = 0; hS[1][1] = -1; hS[1][2] = 0;
            hS[2][0] = 0; hS[2][1] = 1; hS[2][2] = 0;

            hE[0][0] = 0; hE[0][1] = 0; hE[0][2] = 0;
            hE[1][0] = 0; hE[1][1] = -1; hE[1][2] = 1;
            hE[2][0] = 0; hE[2][1] = 0; hE[2][2] = 0;

            hW[0][0] = 0; hW[0][1] = 0; hW[0][2] = 0;
            hW[1][0] = 1; hW[1][1] = -1; hW[1][2] = 0;
            hW[2][0] = 0; hW[2][1] = 0; hW[2][2] = 0;

            hNE[0][0] = 0; hNE[0][1] = 0; hNE[0][2] = 1;
            hNE[1][0] = 0; hNE[1][1] = -1; hNE[1][2] = 0;
            hNE[2][0] = 0; hNE[2][1] = 0; hNE[2][2] = 0;

            hSE[0][0] = 0; hSE[0][1] = 0; hSE[0][2] = 0;
            hSE[1][0] = 0; hSE[1][1] = -1; hSE[1][2] = 0;
            hSE[2][0] = 0; hSE[2][1] = 0; hSE[2][2] = 1;

            hSW[0][0] = 0; hSW[0][1] = 0; hSW[0][2] = 0;
            hSW[1][0] = 0; hSW[1][1] = -1; hSW[1][2] = 0;
            hSW[2][0] = 1; hSW[2][1] = 0; hSW[2][2] = 0;

            hNW[0][0] = 1; hNW[0][1] = 0; hNW[0][2] = 0;
            hNW[1][0] = 0; hNW[1][1] = -1; hNW[1][2] = 0;
            hNW[2][0] = 0; hNW[2][1] = 0; hNW[2][2] = 0;
            #endregion

            for (int t = 0; t < num_iter; t++)
            {
                nablaN = conv(diff_im, width, height, hN, 3);
                nablaS = conv(diff_im, width, height, hS, 3);
                nablaW = conv(diff_im, width, height, hW, 3);
                nablaE = conv(diff_im, width, height, hE, 3);
                nablaNE = conv(diff_im, width, height, hNE, 3);
                nablaSE = conv(diff_im, width, height, hSE, 3);
                nablaSW = conv(diff_im, width, height, hSW, 3);
                nablaNW = conv(diff_im, width, height, hNW, 3);

                if (option == 1)
                {
                    cN = diffusion_function_1(nablaN, kappa, 2, width, height);
                    cS = diffusion_function_1(nablaS, kappa, 2, width, height);
                    cW = diffusion_function_1(nablaW, kappa, 2, width, height);
                    cE = diffusion_function_1(nablaE, kappa, 2, width, height);
                    cNE = diffusion_function_1(nablaNE, kappa, 2, width, height);
                    cSE = diffusion_function_1(nablaSE, kappa, 2, width, height);
                    cSW = diffusion_function_1(nablaSW, kappa, 2, width, height);
                    cNW = diffusion_function_1(nablaNW, kappa, 2, width, height);
                }
                else
                {
                    cN = diffusion_function_2(nablaN, kappa, 2, width, height);
                    cS = diffusion_function_2(nablaS, kappa, 2, width, height);
                    cW = diffusion_function_2(nablaW, kappa, 2, width, height);
                    cE = diffusion_function_2(nablaE, kappa, 2, width, height);
                    cNE = diffusion_function_2(nablaNE, kappa, 2, width, height);
                    cSE = diffusion_function_2(nablaSE, kappa, 2, width, height);
                    cSW = diffusion_function_2(nablaSW, kappa, 2, width, height);
                    cNW = diffusion_function_2(nablaNW, kappa, 2, width, height);
                }
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        diff_im[i][j] = diff_im[i][j] + delta_t * ((1 / Math.Pow(dy, 2)) * cN[i][j] * nablaN[i][j] + (1 / Math.Pow(dy, 2)) * cS[i][j] * nablaS[i][j] + (1 / Math.Pow(dx, 2)) * cW[i][j] * nablaW[i][j] + (1 / Math.Pow(dx, 2)) * cE[i][j] * nablaE[i][j] + (1 / Math.Pow(dd, 2)) * cNE[i][j] * nablaNE[i][j] + (1 / Math.Pow(dd, 2)) * cSE[i][j] * nablaSE[i][j] + (1 / Math.Pow(dd, 2)) * cSW[i][j] * nablaSW[i][j] + (1 / Math.Pow(dd, 2)) * cNW[i][j] * nablaNW[i][j]);
                    }
                }
            }
            double max_value = 0;
            double min_value = 100000;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (diff_im[i][j] > max_value)
                    {
                        max_value = diff_im[i][j];
                    }
                    if (diff_im[i][j] < min_value)
                    {
                        min_value = diff_im[i][j];
                    }
                }
            }

            short[][] rmat = InitialMat(width, height);
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    rmat[i][j] = (short)Math.Round(255 * (diff_im[i][j] - min_value) / (max_value - min_value));
                }
            }
            Bitmap result = mat2Img(rmat, width, height);
            return result;
        }

        #region smoothing
        #endregion

        public Bitmap median(Bitmap img, int filterSize)
        {
            short[] array = new short[filterSize * filterSize];
            Bitmap result = (Bitmap)img.Clone();
            short[][] mat = Img2mat(img);
            short[][] rm = Img2mat(img);
            int limit = filterSize / 2;
            int x, y;
            int count;

            x = limit;
            y = limit;


            while (x < img.Width - limit)
            {
                y = limit;
                while (y < img.Height - limit)
                {
                    count = 0;
                    for (int i = -limit; i <= limit; i++)
                    {
                        for (int j = -limit; j <= limit; j++)
                        {
                            array[count] = mat[x + i][y + j];
                            count = count + 1;
                        }
                    }
                    Array.Sort(array);
                    rm[x][y] = array[((filterSize * filterSize) / 2) + 1];
                    y = y + 1;
                }
                x = x + 1;
            }
            for (int i = 0; i < img.Width; i++)
            {
                for (int j = 0; j < img.Height; j++)
                {
                    rm[i][j] = mat[i][j];
                }
            }

            result = mat2Img(rm, img.Width, img.Height);
            return result;
        }

        public Bitmap mean(Bitmap img, int filterSize)
        {
            Bitmap result = (Bitmap)img.Clone();
            short[][] mat = Img2mat(img);
            short[][] rm = InitialMat(img.Width, img.Height);
            int limit = filterSize / 2;
            int x, y;
            int count;

            x = limit;
            y = limit;

            while (x < img.Width - limit)
            {
                y = limit;
                while (y < img.Height - limit)
                {
                    count = 0;
                    for (int i = -limit; i <= limit; i++)
                    {
                        for (int j = -limit; j <= limit; j++)
                        {
                            count = mat[x + i][y + j] + count;
                        }
                    }

                    rm[x][y] = (short)(count / (filterSize * filterSize));
                    y = y + 1;
                }
                x = x + 1;
            }
            result = mat2Img(rm, img.Width, img.Height);
            return result;
        }

        public Bitmap GaussianBlur(Bitmap img, double sig, int kernal)
        {
            Bitmap bmp = (Bitmap)img.Clone();
            // create filter with kernel size equal to 11
            // and Gaussia sigma value equal to 4.0
            GaussianBlur filter = new GaussianBlur(sig, kernal);
            // apply the filter
            filter.Apply(bmp);
            return bmp;
        }

        public Bitmap ConservativeBlur(Bitmap img)
        {
            Bitmap bmp;
            bmp = ColorToGrayscalev2(img);
            ConservativeSmoothing filter = new ConservativeSmoothing();
            bmp = filter.Apply(bmp);
            return bmp;
        }

        public Bitmap JitterBlur(Bitmap img, int radius)
        {
            Bitmap bmp = new Bitmap(img.Width, img.Height);
            Jitter filter = new Jitter(radius);
            bmp = ColorToGrayscalev2(img);
            bmp = filter.Apply(bmp);
            return bmp;
        }

        public Bitmap OilPaintBlur(Bitmap img, int window)
        {
            Bitmap bmp = new Bitmap(img.Width, img.Height);
            bmp = ColorToGrayscalev2(img);
            OilPainting filter = new OilPainting(window);
            bmp = filter.Apply(bmp);
            return bmp;
        }

        public Bitmap anisotropic(Bitmap image, double K, double lambda)
        {
            int w = image.Width;
            int h = image.Height;
            short[][] imat = Img2mat(image);
            int[][] Nmat = new int[w][];
            int[][] Smat = new int[w][];
            int[][] Emat = new int[w][];
            int[][] Wmat = new int[w][];
            double[][] CNmat = new double[w][];
            double[][] CSmat = new double[w][];
            double[][] CEmat = new double[w][];
            double[][] CWmat = new double[w][];
            short[][] result = new short[w][];
            for (int i = 0; i < w; i++)
            {
                Nmat[i] = new int[h];
                Smat[i] = new int[h];
                Emat[i] = new int[h];
                Wmat[i] = new int[h];
                CNmat[i] = new double[h];
                CSmat[i] = new double[h];
                CEmat[i] = new double[h];
                CWmat[i] = new double[h];
                result[i] = new short[h];
            }
            for (int i = 1; i < w - 1; i++)
            {
                for (int j = 1; j < h - 1; j++)
                {
                    Nmat[i][j] = imat[i - 1][j] - imat[i][j];
                    Smat[i][j] = imat[i + 1][j] - imat[i][j];
                    Emat[i][j] = imat[i][j + 1] - imat[i][j];
                    Wmat[i][j] = imat[i][j - 1] - imat[i][j];
                    CNmat[i][j] = 1 / (1 + Math.Pow((Math.Abs(Nmat[i][j]) / K), 2));
                    CSmat[i][j] = 1 / (1 + Math.Pow((Math.Abs(Smat[i][j]) / K), 2));
                    CEmat[i][j] = 1 / (1 + Math.Pow((Math.Abs(Emat[i][j]) / K), 2));
                    CWmat[i][j] = 1 / (1 + Math.Pow((Math.Abs(Wmat[i][j]) / K), 2));
                }
            }
            for (int j = 0; j < h; j++)
            {
                result[0][j] = imat[0][j];
                result[w - 1][j] = imat[w - 1][j];
            }
            for (int i = 0; i < w; i++)
            {
                result[i][0] = imat[i][0];
                result[i][h - 1] = imat[i][h - 1];
            }
            for (int i = 1; i < w - 1; i++)
            {
                for (int j = 1; j < h - 1; j++)
                {
                    result[i][j] = (short)Math.Round(imat[i][j] + lambda * (CNmat[i][j] * Nmat[i][j] + CSmat[i][j] * Smat[i][j] + CEmat[i][j] * Emat[i][j] + CWmat[i][j] * Wmat[i][j]));
                    if (result[i][j] > 255)
                        result[i][j] = 255;
                    if (result[i][j] < 0)
                        result[i][j] = 0;
                }
            }
            Bitmap output = mat2Img(result, w, h);
            return output;
        }


        #region Edge detection
        #endregion

        public Bitmap DifferenceEdge(Bitmap img)
        {
            Bitmap bmp;
            // create filter
            DifferenceEdgeDetector filter = new DifferenceEdgeDetector();
            bmp = ColorToGrayscalev2(img);
            filter.ApplyInPlace(bmp);
            Bitmap output = GrayscaleToColor(bmp);
            int threshold = ot.getOtsuThreshold(output);
            output = ot.GetBinaryImg(output, threshold);
            return output;
            //return bmp;
        }

        public Bitmap HomogenityEdge(Bitmap img)
        {
            Bitmap bmp;
            // create filter
            HomogenityEdgeDetector filter = new HomogenityEdgeDetector();
            bmp = ColorToGrayscalev2(img);
            filter.ApplyInPlace(bmp);
            Bitmap output = GrayscaleToColor(bmp);
            return output;
            //return bmp;
        }

        public Bitmap SobelEdge(Bitmap img)
        {
            Bitmap bmp;
            // create filter
            SobelEdgeDetector filter = new SobelEdgeDetector();
            bmp = ColorToGrayscalev2(img);
            filter.ApplyInPlace(bmp);
            Bitmap output = GrayscaleToColor(bmp);
            output = setEdgeThreshold(output, 0.8);
            return output;
            //return bmp;
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

        public Bitmap DifferenceEdgeDetector(Bitmap img)
        {
            Bitmap edgeImg;
            int width, height;
            width = img.Width;
            height = img.Height;
            edgeImg = new Bitmap(width, height);
            short[][] imat = Img2mat(img);
            short[][] emat = InitialMat(width, height);
            int[] tempArray = new int[4];
            Console.WriteLine("Difference Edge Detection.");
            for (int i = 1; i < width - 1; i++)
            {
                for (int j = 1; j < height - 1; j++)
                {
                    tempArray[0] = Math.Abs(imat[i - 1][j - 1] - imat[i + 1][j + 1]);
                    tempArray[1] = Math.Abs(imat[i][j - 1] - imat[i][j + 1]);
                    //tempArray[1] = 0;
                    tempArray[2] = Math.Abs(imat[i + 1][j - 1] - imat[i - 1][j + 1]);
                    tempArray[3] = Math.Abs(imat[i + 1][j] - imat[i - 1][j]);
                    Array.Sort(tempArray);
                    emat[i][j] = (short)tempArray[3];
                }
            }
            edgeImg = mat2Img(emat, width, height);
            return edgeImg;
        }

        public Bitmap LapacianEdgeDetector(Bitmap img)
        {
            Bitmap edgeImg;
            int width, height;
            width = img.Width;
            height = img.Height;
            edgeImg = new Bitmap(width, height);
            short[][] imat = Img2mat(img);
            short[][] emat = InitialMat(width, height);

            int pmin = 255;
            int pmax = -255;
            int temp;

            for (int i = 1; i < width - 1; i++)
            {
                for (int j = 1; j < height - 1; j++)
                {
                    temp = -(imat[i - 1][j - 1] + imat[i][j - 1] + imat[i + 1][j - 1] + imat[i - 1][j] + imat[i + 1][j] + imat[i - 1][j + 1] + imat[i][j + 1] + imat[i + 1][j + 1]) + (8 * imat[i][j]);
                    emat[i][j] = (short)temp;
                    if (temp < pmin)
                    {
                        pmin = temp;
                    }
                    if (temp > pmax)
                    {
                        pmax = temp;
                    }
                }
            }
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    //emat[i][j] = (short)(255 * ((double)(emat[i][j] - pmin) / (pmax - pmin)));
                    if (emat[i][j] > 255)
                        emat[i][j] = 255;
                    if (emat[i][j] < 0)
                        emat[i][j] = 0;
                }
            }
            edgeImg = mat2Img(emat, width, height);
            return edgeImg;
        }

        public Bitmap SobelEdgeDetector(Bitmap image, bool verticle)
        {
            int[][] mask = new int[3][];

            for (int i = 0; i < 3; i++)
            {
                mask[i] = new int[3];
            }
            if (verticle)
            {
                mask[0][0] = -1;
                mask[0][1] = -2;
                mask[0][2] = -1;
                mask[1][0] = 0;
                mask[1][1] = 0;
                mask[1][2] = 0;
                mask[2][0] = 1;
                mask[2][1] = 2;
                mask[2][2] = 1;
            }
            else
            {
                mask[0][0] = -1;
                mask[0][1] = 0;
                mask[0][2] = 1;
                mask[1][0] = -2;
                mask[1][1] = 0;
                mask[1][2] = 2;
                mask[2][0] = -1;
                mask[2][1] = 0;
                mask[2][2] = 1;
            }
            short[][] imat = Img2mat(image);
            short[][] rmat = InitialMat(image.Width, image.Height);
            int x, y;
            int width = image.Width;
            int height = image.Height;
            for (int i = 1; i < width - 1; i++)
            {
                for (int j = 1; j < height - 1; j++)
                {
                    int tmp = 0;
                    x = -1;
                    while (x < 2)
                    {
                        y = -1;
                        while (y < 2)
                        {
                            tmp = tmp + (mask[x + 1][y + 1] * imat[i + x][j + y]);
                            y++;
                        }
                        x++;
                    }


                    tmp = Convert.ToInt32(Math.Sqrt(Math.Pow(tmp, 2)));
                    //if (tmp > 255)
                    //{ tmp = 255; }
                    //if(tmp<0)
                    //{ tmp = 0; }
                    rmat[i][j] = (short)tmp;
                }
            }
            //int minvalue = 255;
            int maxvalue = 0;
            //for (int i = 0; i < image.Width; i++)
            //{
            //    for (int j = 0; j < image.Height; j++)
            //    {
            //        if (rmat[i][j] > maxvalue)
            //        {
            //            maxvalue = rmat[i][j];
            //        }
            //        if (rmat[i][j] < minvalue)
            //        {
            //            minvalue = rmat[i][j];
            //        }
            //    }
            //}
            //for (int i = 0; i < image.Width; i++)
            //{
            //    for (int j = 0; j < image.Height; j++)
            //    {
            //        rmat[i][j] = (short)(255 * (rmat[i][j] - minvalue) / (maxvalue - minvalue));
            //    }
            //}
            //maxvalue = 0;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (rmat[i][j] > maxvalue)
                    {
                        maxvalue = rmat[i][j];
                    }
                }
            }
            maxvalue = (int)Math.Round(maxvalue * 0.25);
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    if (rmat[i][j] < maxvalue)
                    {
                        rmat[i][j] = 0;
                    }
                    else
                    {
                        rmat[i][j] = 255;
                    }
                }
            }
            Bitmap result = mat2Img(rmat, width, height);
            return result;
        }

        #region Labeling
        #endregion

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


        #region Appendix I
        #endregion

        public Bitmap setEdgeThreshold(Bitmap image, double ratio)
        {
            short[][] imat = Img2mat(image);
            int threshold;
            int maxvalue = 0;
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    if (imat[i][j] > maxvalue)
                    {
                        maxvalue = imat[i][j];
                    }
                }
            }
            threshold = (int)(maxvalue * ratio);
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    if (imat[i][j] >= threshold)
                    {
                        imat[i][j] = 255;
                    }
                    else
                    {
                        imat[i][j] = 0;
                    }
                }
            }
            Bitmap result = mat2Img(imat, image.Width, image.Height);
            return result;
        }

        public void closingOperator(ref Bitmap img)
        {
            // create filter
            Closing filter = new Closing();
            // apply the filter
            Bitmap tmp = ColorToGrayscalev2(img);
            filter.ApplyInPlace(tmp);
            img = GrayscaleToColor(tmp);

            //// create filter
            //Closing filter = new Closing();
            //// apply the filter
            //filter.ApplyInPlace(img);

        }

        public void openingOperator(ref Bitmap img)
        {
            // Create filter
            Opening filter = new Opening();
            // apply the filter
            Bitmap tmp = ColorToGrayscalev2(img);
            filter.Apply(tmp);
            img = GrayscaleToColor(tmp);
        }

        public Bitmap openingfilter(Bitmap img, int filterSize)
        {
            Bitmap result = img.DilateAndErodeFilter(filterSize, ExtBitmap.MorphologyType.Erosion, true, true, true);
            result = result.DilateAndErodeFilter(filterSize, ExtBitmap.MorphologyType.Dilation, true, true, true);
            return result;
        }

        public Bitmap closingfilter(Bitmap img, int filterSize)
        {
            Bitmap result = img.DilateAndErodeFilter(filterSize, ExtBitmap.MorphologyType.Dilation, true, true, true);
            result = result.DilateAndErodeFilter(filterSize, ExtBitmap.MorphologyType.Erosion, true, true, true);
            return result;
        }

        public Bitmap cornerDetect(Bitmap img)
        {
            /*
            // create corner detector's instance
            SusanCornersDetector scd = new SusanCornersDetector();
            // create corner maker filter
            CornersMarker filter = new CornersMarker(scd, Color.Red);
            // apply the filter
            Bitmap bmp = ColorToGrayscalev2(img);
            filter.ApplyInPlace(bmp);
            */
            // create corners detector's instance
            SusanCornersDetector scd = new SusanCornersDetector();
            Bitmap bmp = ColorToGrayscalev2(img);
            // process image searching for corners
            List<IntPoint> corners = scd.ProcessImage(bmp);
            Bitmap output = GrayscaleToColor(bmp);
            // process points
            foreach (IntPoint corner in corners)
            {
                // ... 
                output.SetPixel(corner.X, corner.Y, Color.Red);
            }

            return output;
        }

        public short[][] divergence(int[][] mat, int width, int height)
        {
            short[][] div = InitialMat(width, height);
            short[][] px = InitialMat(width, height);
            short[][] py = InitialMat(width, height);
            for (int i = 0; i < width - 1; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    px[i][j] = (short)(mat[i + 1][j] - mat[i][j]);
                }

            }
            for (int j = 0; j < height - 1; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    py[i][j] = (short)(mat[i][j + 1] - mat[i][j]);
                }
            }
            for (int i = 0; i < width; i++)
            {
                py[i][height - 1] = (short)mat[i][height - 1];
            }
            for (int j = 0; j < height; j++)
            {
                py[width - 1][j] = (short)mat[width - 1][j];
            }
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    div[i][j] = (short)(px[i][j] + py[i][j]);
                }
            }
            return div;
        }

        public double norm(short[][] imat, int width, int height)
        {
            double normValue = 0;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    normValue = normValue + Math.Abs(imat[i][j]);
                }
            }
            return normValue;
        }

        public double[][] conv(double[][] imat, int width, int height, short[][] mask, int maskSize)
        {
            int subSize = maskSize / 2;
            double[][] conv_mat = new double[width][];
            for (int i = 0; i < width; i++)
            {
                conv_mat[i] = new double[height];
                for (int j = 0; j < height; j++)
                {
                    conv_mat[i][j] = imat[i][j];
                }
            }
            double tmp = 0;
            for (int i = subSize; i < width - subSize; i++)
            {
                for (int j = subSize; j < height - subSize; j++)
                {
                    int x = -subSize;
                    tmp = 0;
                    while (x <= subSize)
                    {
                        int y = -subSize;
                        while (y <= subSize)
                        {
                            tmp = tmp + (imat[i + x][j + y] * mask[x + subSize][y + subSize]);
                            y++;
                        }
                        x++;
                    }
                    conv_mat[i][j] = tmp;
                }
            }
            return conv_mat;
        }

        public double[][] matrix_DIV_constant(double[][] A, double value, int width, int height)
        {
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    A[i][j] = A[i][j] / value;
                }
            }
            return A;
        }

        public double[][] matrix_Power(double[][] A, int width, int height, double power_para)
        {
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    A[i][j] = Math.Pow(A[i][j], power_para);
                }
            }
            return A;
        }

        public double[][] matrix_exp(double[][] A, int width, int height)
        {
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    A[i][j] = Math.Exp(A[i][j]);
                }
            }
            return A;
        }

        #region Appendix II
        #endregion

        public Bitmap inverseImage(Bitmap image)
        {
            short[][] imat = Img2mat(image);
            Bitmap result = new Bitmap(image.Width, image.Height);
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    if (imat[i][j] == 255)
                    {
                        result.SetPixel(i, j, Color.Black);
                    }
                    if (imat[i][j] == 0)
                    {
                        result.SetPixel(i, j, Color.White);
                    }
                }
            }
            return result;

        }

        public Bitmap filterblob_size(Bitmap image, Rectangle[] rects, int size)
        {
            // filter blob only using blob size and filter the candidate blob around the center line
            short[][] imat = Img2mat(image);
            int ideal_center = image.Width / 2;
            int[] blob = new int[rects.Length];
            for (int i = 0; i < rects.Length; i++)
            {
                int blobSize = 0;
                int x = rects[i].X;
                int y = rects[i].Y;
                int bw = rects[i].Width;
                int bh = rects[i].Height;
                int index_x = x;
                int center = 0;
                while (index_x < x + bw)
                {
                    int index_y = y;
                    while (index_y < y + bh)
                    {
                        if (imat[index_x][index_y] != 0)
                        {
                            blobSize = blobSize + 1;
                            center = center + index_x;
                        }
                        index_y = index_y + 1;
                    }
                    index_x = index_x + 1;
                }
                if (blobSize > 0)
                {
                    center = center / blobSize;
                }
                blob[i] = blobSize;
                if ((blobSize < size) | Math.Abs(center - image.Width / 2) < 30)
                //if (blobSize < size)
                {
                    index_x = x;
                    while (index_x < x + bw)
                    {
                        int index_y = y;
                        while (index_y < y + bh)
                        {
                            if (imat[index_x][index_y] == 255)
                            {
                                imat[index_x][index_y] = 0;
                            }
                            index_y = index_y + 1;
                        }
                        index_x = index_x + 1;
                    }
                }
            }
            image = mat2Img(imat, image.Width, image.Height);
            return image;
        }

        public System.Drawing.Point[] centralPoint(Bitmap image, Rectangle[] rects)
        {
            System.Drawing.Point[] cp = new System.Drawing.Point[rects.Length];
            short[][] imat = Img2mat(image);
            for (int i = 0; i < rects.Length; i++)
            {
                int x = rects[i].X;
                int y = rects[i].Y;
                int bw = rects[i].Width;
                int bh = rects[i].Height;
                int index_x = x;
                int blob_index = 0;
                int center_x = 0;
                int center_y = 0;
                while (index_x < x + bw)
                {
                    int index_y = rects[i].Y;
                    while (index_y < y + bh)
                    {
                        if (imat[index_x][index_y] != 0)
                        {
                            blob_index = blob_index + 1;
                            center_x = center_x + index_x;
                            center_y = center_y + index_y;
                        }
                        index_y = index_y + 1;
                    }
                    index_x = index_x + 1;
                }
                center_x = center_x / blob_index;
                center_y = center_y / blob_index;
                //int center_x = (rects[i].X + rects[i].Width) / 2;
                //int center_y = (rects[i].Y + rects[i].Height) / 2;
                cp[i].X = center_x;
                cp[i].Y = center_y;
            }
            return cp;

        }


        #region Pre-process Algorithm
        #endregion

        public int getbackroundThreshold(Bitmap img)
        {
            int w = img.Width;
            int h = img.Height;
            int threshold = 0;
            short[][] imat = Img2mat(img);
            //double tmp1 = 0;
            double tmp2 = 0;
            //for(int i=0;i<50;i++)
            //{
            //    for(int j=0;j<50;j++)
            //    {
            //        tmp1 = tmp1 + imat[i][j];
            //    }
            //}
            //tmp1 = tmp1 / 2500;
            for (int i = w - 1 - 50; i < w; i++)
            {
                for (int j = 0; j < 50; j++)
                {
                    tmp2 = tmp2 + imat[i][j];
                }
            }
            tmp2 = tmp2 / 2500;
            threshold = (int)Math.Round(tmp2);
            //threshold = (int)Math.Ceiling((tmp1 + tmp2) / 2);
            return threshold;
        }

        public Bitmap closeTemplate(Bitmap image)
        {
            image = inverseImage(image);
            //Closing filter = new Closing();
            Dilatation filter = new Dilatation();
            Bitmap tmp = ColorToGrayscalev2(image);
            filter.ApplyInPlace(tmp);

            Bitmap result = GrayscaleToColor(tmp);
            result = inverseImage(result);
            return result;
        }

        public double[][] diffusion_function_1(double[][] A, double kappa, double power, int width, int height)
        {
            double[][] tmp = new double[width][];
            for (int i = 0; i < width; i++)
            {
                tmp[i] = new double[height];
            }
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    tmp[i][j] = Math.Exp(-Math.Pow(A[i][j] / kappa, power));
                }
            }
            return tmp;
        }

        public double[][] diffusion_function_2(double[][] A, double kappa, double power, int width, int height)
        {
            double[][] tmp = new double[width][];
            for (int i = 0; i < width; i++)
            {
                tmp[i] = new double[height];
                for (int j = 0; j < height; j++)
                {
                    tmp[i][j] = 1 / (1 + Math.Pow(A[i][j] / kappa, power));
                }
            }
            return tmp;
        }

        #region Paper X-ray image enhancement
        #endregion

        public Bitmap multiscale_morphologyEnhance(Bitmap image)
        {
            int w = image.Width;
            int h = image.Height;
            int[][] Fo = new int[w][];
            int[][] Fc = new int[w][];
            int[][] open_result = new int[w][];
            int[][] close_result = new int[w][];
            for (int m = 0; m < w; m++)
            {
                Fo[m] = new int[h];
                Fc[m] = new int[h];
                open_result[m] = new int[h];
                close_result[m] = new int[h];
            }
            for (int m = 6; m >= 1; m--)
            {
                Bitmap preimg = (Bitmap)image.Clone();
                Bitmap afimg = (Bitmap)image.Clone();
                //preimg = preimg.OpenMorphologyFilter(2 * m + 1, true, true, true);
                //afimg = afimg.OpenMorphologyFilter(2 * m + 3, true, true, true);
                preimg = preimg.OpenMorphologyFilter(2 * m + 1, true, true, true);
                afimg = afimg.OpenMorphologyFilter(2 * m - 1, true, true, true);
                short[][] pmat = Img2mat(preimg);
                short[][] amat = Img2mat(afimg);
                Bitmap cpreimg = (Bitmap)image.Clone();
                Bitmap cafimg = (Bitmap)image.Clone();
                cpreimg = cpreimg.CloseMorphologyFilter(2 * m + 1, true, true, true);
                cafimg = cafimg.CloseMorphologyFilter(2 * m - 1, true, true, true);
                short[][] cpmat = Img2mat(cpreimg);
                short[][] camat = Img2mat(cafimg);
                for (int a = 0; a < w; a++)
                {
                    for (int b = 0; b < h; b++)
                    {
                        open_result[a][b] = open_result[a][b] + (pmat[a][b] - amat[a][b]);
                        close_result[a][b] = close_result[a][b] + (camat[a][b] - cpmat[a][b]);
                    }
                }
            }
            short[][] imat = Img2mat(image);
            short[][] result = InitialMat(w, h);
            for (int m = 0; m < w; m++)
            {
                for (int n = 0; n < h; n++)
                {
                    result[m][n] = (short)(imat[m][n] + 0.5 * open_result[m][n] - 0.5 * close_result[m][n]);
                    if (result[m][n] > 255)
                        result[m][n] = 255;
                    if (result[m][n] < 0)
                        result[m][n] = 0;
                }
            }
            Bitmap eimage = mat2Img(result, w, h);
            return eimage;
        }

        public Bitmap statisticalOperationEnhance(Bitmap image, int bound)
        {
            int w = image.Width;
            int h = image.Height;
            Bitmap a1 = new Bitmap(w / 2, bound);
            Bitmap a2 = new Bitmap(w / 2, bound);
            Bitmap a3 = new Bitmap(w / 2, h - bound);
            Bitmap a4 = new Bitmap(w / 2, h - bound);
            Bitmap result = (Bitmap)image.Clone();
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    if (i < w / 2 && j < bound)
                    {
                        a1.SetPixel(i, j, image.GetPixel(i, j));
                    }
                    if (i >= w / 2 && j < bound)
                    {
                        a2.SetPixel(i - (w / 2), j, image.GetPixel(i, j));
                    }
                    if (i < w / 2 && j >= bound)
                    {
                        a3.SetPixel(i, j - bound, image.GetPixel(i, j));
                    }
                    if (i >= w / 2 && j >= bound)
                    {
                        a4.SetPixel(i - (w / 2), j - bound, image.GetPixel(i, j));
                    }
                }
            }
            Bitmap ha1 = histEq(a1);
            Bitmap ha2 = histEq(a2);
            Bitmap ha3 = histEq(a3);
            Bitmap ha4 = histEq(a4);
            int x1, x2, x3, x4;
            int m1, m2, m3, m4;
            int tmpmax = 0;
            int tmpmin = 255;
            short[][] mat_a1 = Img2mat(a1);
            short[][] mat_a2 = Img2mat(a2);
            short[][] mat_a3 = Img2mat(a3);
            short[][] mat_a4 = Img2mat(a4);
            m1 = 0;
            for (int i = 0; i < a1.Width; i++)
            {
                for (int j = 0; j < a1.Height; j++)
                {
                    m1 = m1 + mat_a1[i][j];
                    if (tmpmax < mat_a1[i][j])
                        tmpmax = mat_a1[i][j];
                    if (tmpmin > mat_a1[i][j])
                        tmpmin = mat_a1[i][j];
                }
            }
            m1 = m1 / (a1.Width * a1.Height);
            x1 = (tmpmax - tmpmin) / 2;

            m2 = 0;
            tmpmax = 0;
            tmpmin = 255;
            for (int i = 0; i < a2.Width; i++)
            {
                for (int j = 0; j < a2.Height; j++)
                {
                    m2 = m2 + mat_a2[i][j];
                    if (tmpmax < mat_a2[i][j])
                        tmpmax = mat_a2[i][j];
                    if (tmpmin > mat_a2[i][j])
                        tmpmin = mat_a2[i][j];
                }
            }
            m2 = m2 / (a2.Width * a2.Height);
            x2 = (tmpmax - tmpmin) / 2;

            m3 = 0;
            tmpmax = 0;
            tmpmin = 255;
            for (int i = 0; i < a3.Width; i++)
            {
                for (int j = 0; j < a3.Height; j++)
                {
                    m3 = m3 + mat_a3[i][j];
                    if (tmpmax < mat_a3[i][j])
                        tmpmax = mat_a3[i][j];
                    if (tmpmin > mat_a3[i][j])
                        tmpmin = mat_a3[i][j];
                }
            }
            m3 = m3 / (a3.Width * a3.Height);
            x3 = (tmpmax - tmpmin) / 2;

            m4 = 0;
            tmpmax = 0;
            tmpmin = 255;
            for (int i = 0; i < a4.Width; i++)
            {
                for (int j = 0; j < a4.Height; j++)
                {
                    m4 = m4 + mat_a4[i][j];
                    if (tmpmax < mat_a4[i][j])
                        tmpmax = mat_a4[i][j];
                    if (tmpmin > mat_a4[i][j])
                        tmpmin = mat_a4[i][j];
                }
            }
            m4 = m4 / (a4.Width * a4.Height);
            x4 = (tmpmax - tmpmin) / 2;

            int t1 = Math.Abs(x1 - m1);
            int t2 = Math.Abs(x2 - m2);
            int t3 = Math.Abs(x3 - m3);
            int t4 = Math.Abs(x4 - m4);

            int tmpw, tmph;

            tmpw = a1.Width;
            tmph = a1.Height;
            double std;
            double xmean;
            for (int i = 1; i < tmpw - 1; i++)
            {
                for (int j = 1; j < tmph - 1; j++)
                {
                    std = 0;
                    xmean = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            xmean = xmean + mat_a1[i + m][j + n];
                        }
                    }
                    xmean = xmean / 9;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {

                            std = std + Math.Pow(mat_a1[i + m][j + n] - xmean, 2);
                        }
                    }
                    std = Math.Sqrt(std / (tmpw * tmph));
                    int diff = mat_a1[i][j] - (int)Math.Round(std);
                    if (diff > t1)
                    {
                        result.SetPixel(i, j, ha1.GetPixel(i, j));
                    }
                }
            }

            tmpw = a2.Width;
            tmph = a2.Height;
            for (int i = 1; i < tmpw - 1; i++)
            {
                for (int j = 1; j < tmph - 1; j++)
                {
                    std = 0;
                    xmean = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            xmean = xmean + mat_a2[i + m][j + n];
                        }
                    }
                    xmean = xmean / 9;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {

                            std = std + Math.Pow(mat_a2[i + m][j + n] - xmean, 2);
                        }
                    }
                    std = Math.Sqrt(std / (tmpw * tmph));
                    int diff = mat_a2[i][j] - (int)Math.Round(std);
                    if (diff > t2)
                    {
                        result.SetPixel(i + (w / 2), j, ha2.GetPixel(i, j));
                    }
                }
            }
            tmpw = a3.Width;
            tmph = a3.Height;
            for (int i = 1; i < tmpw - 1; i++)
            {
                for (int j = 1; j < tmph - 1; j++)
                {
                    std = 0;
                    xmean = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            xmean = xmean + mat_a3[i + m][j + n];
                        }
                    }
                    xmean = xmean / 9;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {

                            std = std + Math.Pow(mat_a3[i + m][j + n] - xmean, 2);
                        }
                    }
                    std = Math.Sqrt(std / (tmpw * tmph));
                    int diff = mat_a3[i][j] - (int)Math.Round(std);
                    if (diff > t3)
                    {
                        result.SetPixel(i, j + bound, ha3.GetPixel(i, j));
                    }
                }
            }
            tmpw = a4.Width;
            tmph = a4.Height;
            for (int i = 1; i < tmpw - 1; i++)
            {
                for (int j = 1; j < tmph - 1; j++)
                {
                    std = 0;
                    xmean = 0;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {
                            xmean = xmean + mat_a4[i + m][j + n];
                        }
                    }
                    xmean = xmean / 9;
                    for (int m = -1; m <= 1; m++)
                    {
                        for (int n = -1; n <= 1; n++)
                        {

                            std = std + Math.Pow(mat_a4[i + m][j + n] - xmean, 2);
                        }
                    }
                    std = Math.Sqrt(std / (tmpw * tmph));
                    int diff = mat_a4[i][j] - (int)Math.Round(std);
                    if (diff > t4)
                    {
                        result.SetPixel(i + (w / 2), j + bound, ha4.GetPixel(i, j));
                    }
                }
            }
            return result;
        }

        #region resize image
        #endregion

        public Bitmap ResizeBitmap(Bitmap sourceBMP, int width, int height)
        {
            Bitmap result = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(result))
                g.DrawImage(sourceBMP, 0, 0, width, height);
            return result;
        }

        #region Hough Transform (Circle detection)
        #endregion

        public Bitmap circleDetection(Bitmap image)
        {
            Image<Bgr, Byte> img = new Image<Bgr, Byte>(image);
            Image<Gray, Byte> cannyEdges = img.Convert<Gray, Byte>().PyrDown().PyrUp();
            Gray cannyThreshold = new Gray(30);
            Gray cannyThresholdLinking = new Gray(10);
            Gray circleAccumulatorThreshold = new Gray(30);
            int minRadius = (int)(image.Height * 0.1*0.5);
            int maxRadius = (int)(image.Width * 0.1*0.5);
            if (minRadius > maxRadius)
            {
                int tmp = minRadius;
                minRadius = maxRadius;
                maxRadius = tmp;
            }
            Console.WriteLine("max radius: {0}, min radius: {1}", maxRadius, minRadius);

            CircleF[] circles = cannyEdges.HoughCircles(
                cannyThreshold,
                circleAccumulatorThreshold,
                1.0, //Resolution of the accumulator used to detect centers of the circles
                10, //min distance 
                minRadius, //min radius
                maxRadius //max radius
                )[0]; //Get the circles from the first channel

            #region draw circles
            Image<Bgr, Byte> circleImage = img.CopyBlank();
            foreach (CircleF circle in circles)
                circleImage.Draw(circle, new Bgr(Color.YellowGreen), 2);
            Bitmap result = circleImage.ToBitmap();
            return result;
            #endregion

        }

        public Bitmap circleDetection2(Bitmap image, bool filterBlob)
        {

            // locate objects using blob counter
            BlobCounter blobCounter = new BlobCounter();
            SimpleShapeChecker shapeChecker = new SimpleShapeChecker();
            Bitmap otsuimage = ot.GetBinaryImg(image, ot.getOtsuThreshold(image));
            if (filterBlob)
            {
                blobCounter.MinHeight = (int)(image.Height *0.7);
                blobCounter.MinWidth = (int)(image.Width * 0.8);
            }

            blobCounter.ProcessImage(image);
            Blob[] blobs = blobCounter.GetObjectsInformation();
            // create Graphics object to draw on the image and a pen
            Graphics g = Graphics.FromImage(image);
            Pen redPen = new Pen(Color.Red, 2);
            // check each object and draw circle around objects, which
            // are recognized as circles
            for (int i = 0, n = blobs.Length; i < n; i++)
            {
                List<IntPoint> edgePoints = blobCounter.GetBlobsEdgePoints(blobs[i]);

                AForge.Point center;
                float radius;

                if (shapeChecker.IsCircle(edgePoints, out center, out radius))
                {
                    if (radius > image.Height * 0.8 * 0.5)
                    {
                        g.DrawEllipse(redPen, (int)(center.X - radius), (int)(center.Y - radius), (int)(radius * 2), (int)(radius * 2));
                    }
                    
                }
            }

            redPen.Dispose();
            g.Dispose();
            return image;
        }

        public Bitmap circleDetection3(Bitmap image)
        {
            HoughCircleTransformation circleTransform = new HoughCircleTransformation(image.Width/2);
            Bitmap image8dpp = ColorToGrayscalev2(image);
            // apply Hough circle transform
            circleTransform.ProcessImage(image8dpp);
            Bitmap houghCirlceImage = circleTransform.ToBitmap();
            // get circles using relative intensity
            HoughCircle[] circles = circleTransform.GetCirclesByRelativeIntensity(0.8);

            int perfectRIndex = 0;
            int minRadio = image.Width;
            for (int i = 0; i < circles.Length; i++)
            {
                if (circles[i].Radius < minRadio)
                {
                    minRadio = circles[i].Radius;
                    perfectRIndex = i;
                }
            }
            Console.WriteLine("min radius: {0}, perfect circle index: {1}", minRadio, perfectRIndex);
            Graphics g = Graphics.FromImage(image);
            Pen yellowPen = new Pen(Color.Yellow, 2);
            g.DrawEllipse(yellowPen, 
                                    (int)(circles[perfectRIndex].X - circles[perfectRIndex].Radius), 
                                    (int)(circles[perfectRIndex].Y - circles[perfectRIndex].Radius), 
                                    (int)(circles[perfectRIndex].Radius * 2), 
                                    (int)(circles[perfectRIndex].Radius * 2));
//            foreach (HoughCircle circle in circles)
//            {
//                if (circle.Radius > image.Width / 3)
//                {
//                    g.DrawEllipse(yellowPen,
//                                           (int)(circle.X - circle.Radius),
//                                           (int)(circle.Y - circle.Radius),
//                                           (int)(circle.Radius * 2),
//                                           (int)(circle.Radius * 2));
//                }
                
                // ...
//            }
            return image;
        }

        public void circleDetection4(ref Bitmap img,ref CircleF trueCircle)
        {
            appedix aped = new appedix();
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
            int lowThreshold = aped.stdThreshold(cover_img);
            cover_img = aped.removeLowIntPixel(cover_img, lowThreshold);
            Console.WriteLine("max radius: {0}, min radius: {1}", maxRadius, minRadius);
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
                    Console.WriteLine("Best index: {0}", index);
                    break;
                }
            }

            //Draw circles on image
            Image<Bgr, Byte> circleImage = new Image<Bgr, Byte>(img);
            trueCircle = circles[0][index];
            circleImage.Draw(trueCircle, new Bgr(Color.Tomato), 2);
            img = circleImage.ToBitmap();
            img.Save(@"C:\Users\KR7110\Documents\TestData\testFile\Result\circleTemplate.jpg");
//            Bitmap result = circleImage.ToBitmap();
//            return result;
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
                        if (mat[i,j] > 0)
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


    }

}

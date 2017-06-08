using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
//using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Markup;

namespace CellMax_Circle_Detection
{
    class loadTIFFimage
    {
        public Bitmap loadFromFile(string path)
        {
            Bitmap bmp;
            FileStream fs = new FileStream(path, FileMode.Open);
            bmp = new Bitmap(fs);
            int width = bmp.Width;
            int height = bmp.Height;
            // Get access to the bitmap bits
            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);
            ushort[,] bmpwords = new ushort[width, height];
            unsafe
            {
                ushort* ptr = (ushort*)bd.Scan0;
                for (int iy = 0; iy < height; ++iy)
                {
                    for (int ix = 0; ix < width; ++ix)
                    {
                        bmpwords[iy, ix] = *(ptr + ix);
                    }
                    ptr += bd.Stride / 2;   // NOTE: /2 because we're accessing words!
                }
            }
            Bitmap image = new Bitmap(width, height);
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j > height; j++)
                {
                    image.SetPixel(i, j, Color.FromArgb(bmpwords[i, j], bmpwords[i, j], bmpwords[i, j]));
                }
            }
            return image;
        }

        public Bitmap ConvertTo16bpp(Image img)
        {
            var bmp = new Bitmap(img.Width, img.Height,
                          System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            using (var gr = Graphics.FromImage(bmp))
                gr.DrawImage(img, new Rectangle(0, 0, img.Width, img.Height));
            return bmp;
        }

        public Bitmap ConvertTo24bpp(Bitmap img)
        {
            var bmp = new Bitmap(img.Width, img.Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            using (var gr = Graphics.FromImage(bmp))
                gr.DrawImage(img, new Rectangle(0, 0, img.Width, img.Height));
            return bmp;
        }

        public System.Windows.Controls.Image Bitmap2Image(System.Drawing.Bitmap Bi)
        {
            MemoryStream ms = new MemoryStream();
            Bi.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
            BitmapImage bImage = new BitmapImage();
            bImage.BeginInit();
            bImage.StreamSource = new MemoryStream(ms.ToArray());
            bImage.EndInit();
            ms.Dispose();
            Bi.Dispose();
            System.Windows.Controls.Image i = new System.Windows.Controls.Image();
            i.Source = bImage;
            return i;
        }

        public Image ConvertToImage(System.Windows.Controls.Image img)
        {
            MemoryStream ms = new MemoryStream();
            System.Windows.Media.Imaging.BmpBitmapEncoder bbe = new BmpBitmapEncoder();
            bbe.Frames.Add(BitmapFrame.Create(new Uri(img.Source.ToString(), UriKind.RelativeOrAbsolute)));

            bbe.Save(ms);
            System.Drawing.Image img2 = System.Drawing.Image.FromStream(ms);
            return img2;
        }

        public Bitmap ConvertTo24bpp_Test(Bitmap image)
        {
            Bitmap bmp = new Bitmap(image.Width, image.Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            int width = image.Width;
            int height = image.Height;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    bmp.SetPixel(i, j, Color.FromArgb(image.GetPixel(i, j).R,image.GetPixel(i,j).G,image.GetPixel(i,j).B));
                }
            }
            return bmp;
        }

    }
}

using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Markup;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;
using BitMiracle.LibTiff;
using BitMiracle.LibTiff.Classic;
using System.Diagnostics;

namespace CTC_Preprocessing
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            // Load the file name of raw image for resizing in order to the following process.

            if (folderBrowserDialog1.ShowDialog() == DialogResult.OK)
            {
                int foldNum = 0;
                int seperate = 0;

                textBox1.Text = folderBrowserDialog1.SelectedPath;
                rootDir = textBox1.Text;
                //saveResultDir = rootDir + "\\" + saveResultFolder + "\\";
                subject_Dir = Directory.GetDirectories(@rootDir, "*", SearchOption.TopDirectoryOnly);
                string[] tmpDir;
                string[] tmpfile;
                foreach (string dir in subject_Dir)
                {
                    subjectName.Add(dir.Remove(0, rootDir.Length + 1));
                    //                    Console.WriteLine(dir);
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
                        //if (string.Compare(tmpfile[0], "DAPI Selection", true) == 0)
                        if (tmpfile[0].StartsWith("DAPI Selection"))
                        {
                            fileName_B.Add(filetmp);
                            subject_B_Dir.Add(file);
                        }
                        //if (string.Compare(tmpfile[0], "FITC Selection", true) == 0)
                        if (tmpfile[0].StartsWith("FITC Selection"))
                        {
                            subject_G_Dir.Add(file);
                            fileName_G.Add(filetmp);
                        }
                        //if (string.Compare(tmpfile[0], "TRITC-Rhoadmine Selection", true) == 0)
                        if (tmpfile[0].StartsWith("TRITC-Rhoadmine Selection"))
                        {
                            subject_R_Dir.Add(file);
                            fileName_R.Add(filetmp);
                            //                            Console.WriteLine(filetmp);
                        }
                    }
                    foldNum += 1;
                }
                if ((subject_B_Dir.Count != subject_G_Dir.Count) || (subject_G_Dir.Count != subject_R_Dir.Count) || (subject_B_Dir.Count != subject_R_Dir.Count))
                {
                    Console.WriteLine("Files does not complete.");
                }
                if (foldNum < 1)
                {
                    Console.WriteLine("No folder input for processing...");
                }
                else
                    Console.WriteLine("Continued...");
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            string subjectFolder = "";
            if (folderBrowserDialog1.ShowDialog() == DialogResult.OK)
            {
                textBox2.Text = folderBrowserDialog1.SelectedPath;
                saveResultDir = textBox2.Text;
                zipSaveDir = saveResultDir + "\\zip patient data\\";
                if (Directory.Exists(zipSaveDir))
                {
                    //...
                }
                else
                {
                    //新增資料夾
                    Directory.CreateDirectory(@zipSaveDir);
                }
                for (int i = 0; i < subject_B_Dir.Count; i++)
                {
                    subjectFolder = saveResultDir + "\\" + subjectName[i];
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
        }

        private void button3_Click(object sender, EventArgs e)
        {
            appendix aped = new appendix();
            Stopwatch sw = new Stopwatch(); //using System.Diagnostics;
            int width, height;
            int totalSubject = subject_R_Dir.Count;
            CircleF[] circle = new CircleF[2];
            circle[0] = new CircleF();
            circle[1] = new CircleF();
            //= new CircleF();
            string newFile;

            string recordTime = saveResultDir + "\\Execute Time.txt";
            StreamWriter Timewriter;
            if (rootDir == null)
            {
                Console.WriteLine("Please choose the input image path.");
                button1.PerformClick();
            }
            if (saveResultDir == null)
            {
                Console.WriteLine("Please choose the save path.");
                //button2_Click(sender, e);
                button2.PerformClick();
            }

            Console.WriteLine("Subject Number: {0}", totalSubject);

            for (int index = 0; index < totalSubject; index++)
            {
                width = 0;
                height = 0;
                Timewriter = new StreamWriter(@recordTime,true);
                sw.Reset();
                sw = Stopwatch.StartNew();

                BatchRawFile[0] = subject_R_Dir[index];
                BatchRawFile[1] = subject_G_Dir[index];
                BatchRawFile[2] = subject_B_Dir[index];

                //Console.WriteLine("Load File: {0}", BatchRawFile[0]);
                //Console.WriteLine("Load File: {0}", BatchRawFile[1]);
                //Console.WriteLine("Load File: {0}", BatchRawFile[2]);

                BatchSaveFile[0] = saveResultDir + "/" + subjectName[index] + "/" + fileName_R[index];
                BatchSaveFile[1] = saveResultDir + "/" + subjectName[index] + "/" + fileName_G[index];
                BatchSaveFile[2] = saveResultDir + "/" + subjectName[index] + "/" + fileName_B[index];

                aped.estimateCircle(subject_R_Dir[index], resizeRatio, ref circle, ref width, ref height, saveResultDir);  // estimate the circle in downsampling dimenstion

                //bool[][] map = aped.generateCircleMap(width, height, resizeRatio, circle);  // generate the bool map with pixel inside the circle set true, else false. (original size)

                Console.WriteLine("Image saving...");
                aped.circle_save_batch(BatchRawFile, BatchSaveFile, resizeRatio, circle);



                //aped.saveNewRawDataBatch(BatchRawFile, BatchSaveFile, map);


                Console.WriteLine("Saving circle parameter...");
                BatchSaveFile[3] = saveResultDir + "/" + subjectName[index] + "/" + "Circle Information" + ".txt";
                Console.WriteLine("Save txt File: {0}", BatchSaveFile[3]);

                StreamWriter writer = new StreamWriter(@BatchSaveFile[3]);
                writer.WriteLine("CenterX_1: {0}", (circle[0].Center.X * resizeRatio).ToString());
                writer.WriteLine("CenterY_1: {0}", (circle[0].Center.Y * resizeRatio).ToString());
                writer.WriteLine("Radius_1: {0}", (circle[0].Radius * resizeRatio).ToString());
                writer.WriteLine("CenterX_2: {0}", (circle[1].Center.X * resizeRatio).ToString());
                writer.WriteLine("CenterY_2: {0}", (circle[1].Center.Y * resizeRatio).ToString());
                writer.WriteLine("Radius_2: {0}", (circle[1].Radius * resizeRatio).ToString());
                writer.Close();
                Console.WriteLine("circle information saved.");
                
                newFile = zipSaveDir + subjectName[index] + ".zip";
                aped.compressZIP(BatchSaveFile, newFile);
                sw.Stop();
                Timewriter.WriteLine("Subject: {0}, Time: {1} (min)", subjectName[index], (float)(sw.ElapsedMilliseconds / 1000 / 60.0));
                Timewriter.Close();
                //aped.saveJPG(BatchRawFile[0], saveResultDir, resizeRatio, circle);
                Array.Clear(BatchRawFile, 0, BatchRawFile.Length);
                Array.Clear(BatchSaveFile, 0, BatchSaveFile.Length);
                //Directory.Delete(subject_Dir[index], true);
                File.Delete(newFile);
                newFile = saveResultDir + "/" + subjectName[index];
                Directory.Delete(newFile, true);
            }
            Directory.Delete(zipSaveDir, true);
            Console.WriteLine("THE END. \\^O^/ ");
        }

        private void button5_Click(object sender, EventArgs e)
        {
            string zip_root = @"C:\DataSet\Result\zip patient data\";
            //string zip_root = @"C:\DataSet\Result\123\";
            string[] subject_Dir = Directory.GetFiles(zip_root);
            Console.WriteLine("The number of subjects: {0}", subject_Dir.Length);
            Stopwatch sw = new Stopwatch();
            appendix aped = new appendix();
            sw.Reset();
            sw.Start();
            aped.reCompress(subject_Dir);
            sw.Stop();
            Console.WriteLine("Compress Time: {0} min", (float)(sw.ElapsedMilliseconds / 1000 / 60.0));
            
        }
    }
}

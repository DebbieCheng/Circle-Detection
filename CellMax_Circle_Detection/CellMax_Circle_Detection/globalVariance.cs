using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;

namespace CellMax_Circle_Detection
{
    class globalVariance
    {
    }
    public partial class Form1
    {
        // TRITA: Red channel
        // FITC: Green channel
        // DAPI: Blue channel

        string rootDir;
        string loadDir;
        string saveResultFolder = "Result";
        string test_txt = "";
        string saveResultDir;
        string zipSaveDir;
        string[] subject_Dir;
        string[] file_Dir;
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
        bool cancel_flag = false;
    }
}

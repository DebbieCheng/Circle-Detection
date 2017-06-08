using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.Util;
using Emgu.CV.Structure;

namespace CTC_Preprocessing
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
        //        string saveResultDir = "C:\\Users\\KR7110\\Documents\\TestData\\testFile\\Result\\Result\\Circle-Binary_";
        //string saveResultDir = "C:\\Users\\KR7110\\Documents\\TestData\\testFile\\Result\\RemoveBG_";
        string saveResultDir;
        string zipSaveDir;
        //string zipSaveDir = "C:\\Users\\KR7110\\Documents\\TestData\\testFile\\zip patient data\\";
        string[] subject_Dir;
        string[] file_Dir;
        List<string> subject_R_Dir = new List<string>();
        List<string> subject_G_Dir = new List<string>();
        List<string> subject_B_Dir = new List<string>();
        List<string> fileName_R = new List<string>();
        List<string> fileName_B = new List<string>();
        List<string> fileName_G = new List<string>();
        List<string> subjectName = new List<string>();
        //Rectangle rect = new Rectangle(0, 0, 17122, 15841);
        float resizeRatio = 8;
        loadTIFFimage tif = new loadTIFFimage();
        //List<CircleF> circle_R = new List<CircleF>();
        //List<CircleF> circle_G = new List<CircleF>();
        //List<CircleF> circle_B = new List<CircleF>();
        List<CircleF> circle_BIG = new List<CircleF>();
        string[] BatchRawFile = new string[3];
        string[] BatchSaveFile = new string[4];
    }
}

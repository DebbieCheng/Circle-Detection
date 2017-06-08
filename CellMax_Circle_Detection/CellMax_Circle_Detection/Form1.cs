using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading;

namespace CellMax_Circle_Detection
{
    public partial class Form1 : Form
    {
        private Thread demoThread = null;
        private Thread cancalThread = null;

        public Form1()
        {
            InitializeComponent();
        }
        


        private void button1_Click(object sender, EventArgs e)
        {
            Cancel.InitFlag();
            //if (demoThread!=null)
            //    if (demoThread.IsAlive)
            //        demoThread.Abort();
            demoThread = new Thread(() => ToDetectCircle());
            demoThread.Start();
            //demoThread.Join();
            //circledetection ccr = new circledetection();
            //string root = @"C:\DataSet\rawData";
            //string save = @"C:\DataSet\Result";
            //Console.WriteLine("Start?");
            //ccr.CircleDetection(root, save);
            //Console.WriteLine("Finish.");
        }

        private void ToDetectCircle()
        {
            circledetection ccr = new circledetection();
            string root = @"C:\DataSet\check";
            string save = @"C:\DataSet\check_result";
            Console.WriteLine("Start?");
            ccr.CircleDetection(root, save);
            Console.WriteLine("Finish.");
        }

        private void button2_Click(object sender, EventArgs e)
        {
            //this.Close();
            Cancel.FlagCancel();            
            demoThread.Abort();
            Console.WriteLine("Cancel.");
        }


    }
}

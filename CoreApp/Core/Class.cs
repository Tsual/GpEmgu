using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CoreApp.Core
{
    public class Class
    {

    }
    class FaceRecoginzerTest
    {
        public static void Test()
        {
            using (Mat image = new Mat("testGamePic.jpg"))
            {
                using (Mat uimg = new Mat())
                {
                    using (CascadeClassifier face = new CascadeClassifier("haarcascade_frontalface_default.xml"))
                    {
                        CvInvoke.CvtColor(image, uimg, ColorConversion.Bgr2Gray);
                        CvInvoke.EqualizeHist(uimg, uimg);

                        Rectangle[] facesDetected = face.DetectMultiScale(uimg, 1.1, 10, new Size(20, 20));


                        EigenFaceRecognizer efr = new EigenFaceRecognizer();

                        efr.Train(new VectorOfMat(uimg), new VectorOfInt(new int[] { 1 }));
                        var res = efr.Predict(uimg);


                        return;
                    }

                }
            }
        }






    }
}

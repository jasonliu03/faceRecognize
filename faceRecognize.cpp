//#include "stdafx.h"  
#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <fstream>  
#include <sstream>  
#include <math.h>  
  
using namespace cv;  
using namespace std;  
  
  
int main()   
{  
  
  
    // ÏÂÃæµÄ¼¸ĞĞ´úÂë½ö½öÊÇ´ÓÄãµÄÊı¾İ¼¯ÖĞÒÆ³ı×îºóÒ»ÕÅÍ¼Æ¬  
    //[gm:×ÔÈ»ÕâÀïĞèÒª¸ù¾İ×Ô¼ºµÄĞèÒªĞŞ¸Ä£¬ËûÕâÀï¼ò»¯ÁËºÜ¶àÎÊÌâ]  
    Mat testSample = imread("./att_faces/s40/3.pgm", 0);
    imshow("test", testSample);
    // ÏÂÃæ¼¸ĞĞ´´½¨ÁËÒ»¸öÌØÕ÷Á³Ä£ĞÍÓÃÓÚÈËÁ³Ê¶±ğ£¬  
    // Í¨¹ıCSVÎÄ¼ş¶ÁÈ¡µÄÍ¼ÏñºÍ±êÇ©ÑµÁ·Ëü¡£  
    // TÕâÀïÊÇÒ»¸öÍêÕûµÄPCA±ä»»  
    //Èç¹ûÄãÖ»Ïë±£Áô10¸öÖ÷³É·Ö£¬Ê¹ÓÃÈçÏÂ´úÂë  
    //      cv::createEigenFaceRecognizer(10);  
    //  
    // Èç¹ûÄã»¹Ï£ÍûÊ¹ÓÃÖÃĞÅ¶ÈãĞÖµÀ´³õÊ¼»¯£¬Ê¹ÓÃÒÔÏÂÓï¾ä£º  
    //      cv::createEigenFaceRecognizer(10, 123.0);  
    //  
    // Èç¹ûÄãÊ¹ÓÃËùÓĞÌØÕ÷²¢ÇÒÊ¹ÓÃÒ»¸öãĞÖµ£¬Ê¹ÓÃÒÔÏÂÓï¾ä£º  
    //      cv::createEigenFaceRecognizer(0, 123.0);  
      
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();  
    model->load("MyFacePCAModel.xml");  
  
    Ptr<FaceRecognizer> model1 = createFisherFaceRecognizer();  
    model1->load("MyFaceFisherModel.xml");  
  
    Ptr<FaceRecognizer> model2 = createLBPHFaceRecognizer();  
    model2->load("MyFaceLBPHModel.xml");  
 
    // ÏÂÃæ¶Ô²âÊÔÍ¼Ïñ½øĞĞÔ¤²â£¬predictedLabelÊÇÔ¤²â±êÇ©½á¹û  
    int predictedLabel = model->predict(testSample);  
    int predictedLabel1 = model1->predict(testSample);  
    int predictedLabel2 = model2->predict(testSample);  
  
    // »¹ÓĞÒ»ÖÖµ÷ÓÃ·½Ê½£¬¿ÉÒÔ»ñÈ¡ç??¹ûÍ¬Ê±µÃµ½ãĞÖµ:  
    //      int predictedLabel = -1;  
    //      double confidence = 0.0;  
    //      model->predict(testSample, predictedLabel, confidence);  
      
    cout << predictedLabel << endl;  
    cout << predictedLabel1 << endl;  
    cout << predictedLabel2 << endl;  
  
    waitKey(0);  
    return 0;  
}  

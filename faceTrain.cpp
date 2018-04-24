//#include "stdafx.h"  
#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <fstream>  
#include <sstream>  
#include <math.h>  
  
using namespace cv;  
using namespace std;  
  
static Mat norm_0_255(InputArray _src) {  
    Mat src = _src.getMat();  
    // ´´½¨ºÍ·µ»ØÒ»¸ö¹éÒ»»¯ºóµÄÍ¼Ïñ¾ØÕó:  
    Mat dst;  
    switch (src.channels()) {  
    case1:  
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);  
        break;  
    case3:  
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);  
        break;
    default:
        src.copyTo(dst);  
        break;  
    }  
    return dst;  
}  
  
//Ê¹ÓÃCSVÎÄ¼şÈ¥¶ÁÍ¼ÏñºÍ±êÇ©£¬Ö÷ÒªÊ¹ÓÃstringstreamºÍgetline·½·¨  
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {  
    std::ifstream file(filename.c_str(), ifstream::in);  
    if (!file) {  
        string error_message = "No valid input file was given, please check the given filename.";  
        CV_Error(CV_StsBadArg, error_message);  
    }  
    string line, path, classlabel;  
    while (getline(file, line)) {  
        stringstream liness(line);  
        getline(liness, path, separator);  
        getline(liness, classlabel);  
        if (!path.empty() && !classlabel.empty()) {  
            images.push_back(imread(path, 0));  
            labels.push_back(atoi(classlabel.c_str()));  
        }  
    }  
}  
  
  
int main()   
{  
  
    //¶ÁÈ¡ÄãµÄCSVÎÄ¼şÂ·¾¶.  
    //string fn_csv = string(argv[1]);  
    string fn_csv = "at.txt";  
  
    // 2¸öÈİÆ÷À´´æ·ÅÍ¼ÏñÊı¾İºÍ¶ÔÓ¦µÄ±êÇ©  
    vector<Mat> images;  
    vector<int> labels;  
    // ¶ÁÈ¡Êı¾İ. Èç¹ûÎÄ¼ş²»ºÏ·¨¾Í»á³ö´í  
    // ÊäÈëµÄÎÄ¼şÃûÒÑ¾­ÓĞÁË.  
    try  
    {  
        read_csv(fn_csv, images, labels);  
    }  
    catch (cv::Exception& e)  
    {  
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;  
        // ÎÄ¼şÓĞÎÊÌâ£¬ÎÒÃÇÉ¶Ò²×ö²»ÁËÁË£¬ÍË³öÁË  
        exit(1);  
    }  
    // Èç¹ûÃ»ÓĞ¶ÁÈ¡µ½×ã¹»Í¼Æ¬£¬Ò²ÍË³ö.  
    if (images.size() <= 1) {  
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";  
        CV_Error(CV_StsError, error_message);  
    }  
  
    // ÏÂÃæµÄ¼¸ĞĞ´úÂë½ö½öÊÇ´ÓÄãµÄÊı¾İ¼¯ÖĞÒÆ³ı×îºóÒ»ÕÅÍ¼Æ¬  
    //[gm:×ÔÈ»ÕâÀïĞèÒª¸ù¾İ×Ô¼ºµÄĞèÒªĞŞ¸Ä£¬ËûÕâÀï¼ò»¯ÁËºÜ¶àÎÊÌâ]  
    Mat testSample = images[images.size() - 1];  
    int testLabel = labels[labels.size() - 1];  
    images.pop_back();  
    labels.pop_back();  
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
    model->train(images, labels);  
    model->save("MyFacePCAModel.xml");  
  
    Ptr<FaceRecognizer> model1 = createFisherFaceRecognizer();  
    model1->train(images, labels);  
    model1->save("MyFaceFisherModel.xml");  
  
    Ptr<FaceRecognizer> model2 = createLBPHFaceRecognizer();  
    model2->train(images, labels);  
    model2->save("MyFaceLBPHModel.xml");  
  
    // ÏÂÃæ¶Ô²âÊÔÍ¼Ïñ½øĞĞÔ¤²â£¬predictedLabelÊÇÔ¤²â±êÇ©½á¹û  
    int predictedLabel = model->predict(testSample);  
    int predictedLabel1 = model1->predict(testSample);  
    int predictedLabel2 = model2->predict(testSample);  
  
    // »¹ÓĞÒ»ÖÖµ÷ÓÃ·½Ê½£¬¿ÉÒÔ»ñÈ¡ç??¹ûÍ¬Ê±µÃµ½ãĞÖµ:  
    //      int predictedLabel = -1;  
    //      double confidence = 0.0;  
    //      model->predict(testSample, predictedLabel, confidence);  
      
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);  
    string result_message1 = format("Predicted class = %d / Actual class = %d.", predictedLabel1, testLabel);  
    string result_message2 = format("Predicted class = %d / Actual class = %d.", predictedLabel2, testLabel);  
    cout << result_message << endl;  
    cout << result_message1 << endl;  
    cout << result_message2 << endl;  
  
    waitKey(0);  
    return 0;  
}  

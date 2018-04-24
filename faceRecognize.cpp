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
  
  
    // 下面的几行代码仅仅是从你的数据集中移除最后一张图片  
    //[gm:自然这里需要根据自己的需要修改，他这里简化了很多问题]  
    Mat testSample = imread("./att_faces/s40/3.pgm", 0);
    imshow("test", testSample);
    // 下面几行创建了一个特征脸模型用于人脸识别，  
    // 通过CSV文件读取的图像和标签训练它。  
    // T这里是一个完整的PCA变换  
    //如果你只想保留10个主成分，使用如下代码  
    //      cv::createEigenFaceRecognizer(10);  
    //  
    // 如果你还希望使用置信度阈值来初始化，使用以下语句：  
    //      cv::createEigenFaceRecognizer(10, 123.0);  
    //  
    // 如果你使用所有特征并且使用一个阈值，使用以下语句：  
    //      cv::createEigenFaceRecognizer(0, 123.0);  
      
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();  
    model->load("MyFacePCAModel.xml");  
  
    Ptr<FaceRecognizer> model1 = createFisherFaceRecognizer();  
    model1->load("MyFaceFisherModel.xml");  
  
    Ptr<FaceRecognizer> model2 = createLBPHFaceRecognizer();  
    model2->load("MyFaceLBPHModel.xml");  
 
    // 下面对测试图像进行预测，predictedLabel是预测标签结果  
    int predictedLabel = model->predict(testSample);  
    int predictedLabel1 = model1->predict(testSample);  
    int predictedLabel2 = model2->predict(testSample);  
  
    // 还有一种调用方式，可以获取�??果同时得到阈值:  
    //      int predictedLabel = -1;  
    //      double confidence = 0.0;  
    //      model->predict(testSample, predictedLabel, confidence);  
      
    cout << predictedLabel << endl;  
    cout << predictedLabel1 << endl;  
    cout << predictedLabel2 << endl;  
  
    waitKey(0);  
    return 0;  
}  

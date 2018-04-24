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
  
  
    // ����ļ��д�������Ǵ�������ݼ����Ƴ����һ��ͼƬ  
    //[gm:��Ȼ������Ҫ�����Լ�����Ҫ�޸ģ���������˺ܶ�����]  
    Mat testSample = imread("./att_faces/s40/3.pgm", 0);
    imshow("test", testSample);
    // ���漸�д�����һ��������ģ����������ʶ��  
    // ͨ��CSV�ļ���ȡ��ͼ��ͱ�ǩѵ������  
    // T������һ��������PCA�任  
    //�����ֻ�뱣��10�����ɷ֣�ʹ�����´���  
    //      cv::createEigenFaceRecognizer(10);  
    //  
    // ����㻹ϣ��ʹ�����Ŷ���ֵ����ʼ����ʹ��������䣺  
    //      cv::createEigenFaceRecognizer(10, 123.0);  
    //  
    // �����ʹ��������������ʹ��һ����ֵ��ʹ��������䣺  
    //      cv::createEigenFaceRecognizer(0, 123.0);  
      
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();  
    model->load("MyFacePCAModel.xml");  
  
    Ptr<FaceRecognizer> model1 = createFisherFaceRecognizer();  
    model1->load("MyFaceFisherModel.xml");  
  
    Ptr<FaceRecognizer> model2 = createLBPHFaceRecognizer();  
    model2->load("MyFaceLBPHModel.xml");  
 
    // ����Բ���ͼ�����Ԥ�⣬predictedLabel��Ԥ���ǩ���  
    int predictedLabel = model->predict(testSample);  
    int predictedLabel1 = model1->predict(testSample);  
    int predictedLabel2 = model2->predict(testSample);  
  
    // ����һ�ֵ��÷�ʽ�����Ի�ȡ�??��ͬʱ�õ���ֵ:  
    //      int predictedLabel = -1;  
    //      double confidence = 0.0;  
    //      model->predict(testSample, predictedLabel, confidence);  
      
    cout << predictedLabel << endl;  
    cout << predictedLabel1 << endl;  
    cout << predictedLabel2 << endl;  
  
    waitKey(0);  
    return 0;  
}  

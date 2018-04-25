#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <fstream>  
#include <sstream>  
#include <math.h>  
  
using namespace cv;  
using namespace std;  

/** Global variables */  
String face_cascade_name = "haarcascade_frontalface_default.xml";  
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";  
CascadeClassifier face_cascade;   
CascadeClassifier eyes_cascade;  
String window_name = "Capture - Face detection";  
  
Mat detectAndDisplay(Mat frame)  
{  
    std::vector<Rect> faces;  
    Mat frame_gray;  
  
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);  
    equalizeHist(frame_gray, frame_gray);  
  
    //-- Detect faces  
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, CV_HAAR_DO_ROUGH_SEARCH, Size(50, 50),Size(2000,2000));  

    Mat faceROI = frame_gray;
  
    for (size_t i = 0; i < faces.size(); i++)  
    {  
        rectangle(frame, faces[i],Scalar(255,0,0),2,8,0);  
        cout << faces[i] << endl; 
          
        faceROI = frame_gray(faces[i]);  
        imwrite("faceROI.jpg", faceROI);
    }  
    //-- Show what you got  
    namedWindow(window_name, 2);  
    imshow(window_name, frame);  
    return faceROI;
}  
  
int main(int argc, char** argv)   
{  
    char* imgName = "test.jpg";
    if(argc == 2)
    {
       imgName = argv[1];
    }   
  
    if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };  

    Mat testSample = imread(imgName, 1);
    Mat faceROI = detectAndDisplay(testSample);
    resize(faceROI, faceROI, cvSize(92, 112));
    
      
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();  
    model->load("MyFacePCAModel.xml");  
  
    Ptr<FaceRecognizer> model1 = createFisherFaceRecognizer();  
    model1->load("MyFaceFisherModel.xml");  
  
    Ptr<FaceRecognizer> model2 = createLBPHFaceRecognizer();  
    model2->load("MyFaceLBPHModel.xml");  
 
    int predictedLabel = model->predict(faceROI);  
    int predictedLabel1 = model1->predict(faceROI);  
    int predictedLabel2 = model2->predict(faceROI);  
  
    cout << predictedLabel << endl;  
    cout << predictedLabel1 << endl;  
    cout << predictedLabel2 << endl;  

    int result = -1;
    if(predictedLabel == predictedLabel1)
    {
       result = predictedLabel; 
    }
    if(predictedLabel == predictedLabel2)
    {
       result = predictedLabel; 
    }
    if(predictedLabel1 == predictedLabel2)
    {
       result = predictedLabel1; 
    }

    cout << "result:" << result << endl;
    waitKey(0);  
    return result;  
}  

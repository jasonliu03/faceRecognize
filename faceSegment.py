#!coding=utf-8
import sys
import os
import numpy as np  
import cv2  
  
def faceDetect(imgPath):  
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")  
    #face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")  
    #eye_cascade = cv2.CascadeClassifier("./haarcascade_eye_tree_eyeglasses.xml")  
    
    img = cv2.imread(imgPath)  
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
                          
    faces = face_cascade.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(10,10))  
      
    if len(faces)>0:  
        print "len(faces):", len(faces)
        for faceRect in faces:  
            x,y,w,h = faceRect  
    #        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)  
      
            roi_gray = gray[y:y+h,x:x+w]  
            roi_color = img[y:y+h,x:x+w]  
            
            cv2.imwrite(imgPath, roi_color)
    else:
        print "remove img:", imgPath
        os.remove(imgPath)
      
    #        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,1,cv2.CASCADE_SCALE_IMAGE,(2,2))  
    #        for (ex,ey,ew,eh) in eyes:  
    #            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  
                  
    #cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    #cv2.imshow("img",img)  
    #cv2.waitKey(0) 


if __name__ == "__main__":  
  
    #if len(sys.argv) != 2:  
    #    print "usage: create_csv <base_path>"  
    #    sys.exit(1)  
  
    BASE_PATH="./att_faces"  
    if len(sys.argv) == 2:
        BASE_PATH=sys.argv[1]  
    print "BASE_PATH:", BASE_PATH
        
    print "Note: src dir will be overwritted!, continue? Y or N"
    c = raw_input()
    if c == 'Y':
        pass
    else:
        exit(-1)
      
    for dirname, dirnames, filenames in os.walk(BASE_PATH):  
        if dirnames:
            for subdirname in dirnames:  
                subject_path = os.path.join(dirname, subdirname)  
                for filename in os.listdir(subject_path):  
                    abs_path = "%s/%s" % (subject_path, filename)  
                    print "abs_path:", abs_path
                    faceDetect(abs_path)
        else:
            subject_path = dirname
            for filename in os.listdir(subject_path):  
                abs_path = "%s/%s" % (subject_path, filename)  
                print "abs_path:", abs_path
                faceDetect(abs_path)

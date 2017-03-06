//
// Created by dujiajun on 3/5/17.
//

#include <opencv/cv.hpp>
#include "motion_detector/background_substraction_motion_detector.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
    VideoCapture cap("/home/dujiajun/car_person_video.mp4");
    Mat image;
    BackgroundSubstractionMotionDetector motion_detector;
    while(cap.read(image)){
        vector<Rect> regions = motion_detector.detect(image);
        cout<<"regions num: "<<regions.size()<<endl;
        for(Rect &region: regions){
            rectangle(image, region, Scalar(255, 0, 0), 3, 1);
        }
        imshow("motion detector", image);
        waitKey(100);
    }
}

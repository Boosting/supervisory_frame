//
// Created by dujiajun on 3/5/17.
//

#include "motion_detector/background_substraction_motion_detector.hpp"

BackgroundSubstractionMotionDetector::BackgroundSubstractionMotionDetector(){
    mog = createBackgroundSubtractorMOG2();
}

vector<Rect> BackgroundSubstractionMotionDetector::detect(const Mat &image){
    Mat GaussianImage, foreground;
    GaussianBlur(image, GaussianImage, {11, 11}, 2);
    mog->apply(GaussianImage, foreground, 0.001);
    erode(foreground, foreground, cv::Mat());
    dilate(foreground, foreground, cv::Mat());
}
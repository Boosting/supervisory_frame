//
// Created by dujiajun on 3/5/17.
//

#ifndef SUPERVISORY_FRAME_BACKGROUND_SUBSTRACTION_MOTION_DETECTOR_HPP
#define SUPERVISORY_FRAME_BACKGROUND_SUBSTRACTION_MOTION_DETECTOR_HPP

#include "motion_detector.hpp"

class BackgroundSubstractionMotionDetector{
public:
    BackgroundSubstractionMotionDetector();
    vector<Rect> detect(const Mat &image);
protected:
    Ptr<BackgroundSubtractorMOG2> mog;
    vector<vector<int> > getForegroundMask(const Mat &image);
    vector<Rect> getRegions(vector<vector<int> > &foregroundMask);
    vector<Rect> expandRegions(vector<vector<int> > &foregroundMask, vector<Rect> &regions, vector<Point> &points);
};
#endif //SUPERVISORY_FRAME_BACKGROUND_SUBSTRACTION_MOTION_DETECTOR_HPP

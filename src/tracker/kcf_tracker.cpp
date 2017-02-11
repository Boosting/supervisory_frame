//
// Created by dujiajun on 2/10/17.
//
#include "tracker/kcf_tracker.hpp"
#include "opencv2/tracking.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;
Rect KcfTracker::getUpdateRegion(const Mat& preImage, const Mat& curImage, const Rect& preRegion) {
    //maybe always init one tracker is not efficient
    Ptr<Tracker> tracker = Tracker::create("KCF");
    tracker->init(preImage, Rect2d(preRegion));
    Rect2d curRegion;
    tracker->update(curImage, curRegion);
    return Rect(curRegion);
}

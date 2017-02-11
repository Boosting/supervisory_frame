//
// Created by dujiajun on 2/10/17.
//

#ifndef SUPERVISORY_FRAME_KCF_TRACKER_HPP
#define SUPERVISORY_FRAME_KCF_TRACKER_HPP

#include "class_independent_tracker.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;

class KcfTracker: public ClassIndependentTracker {
public:
    Rect getUpdateRegion(const Mat& preImage, const Mat& curImage, const Rect& preRegion);
};

#endif //SUPERVISORY_FRAME_KCF_TRACKER_HPP

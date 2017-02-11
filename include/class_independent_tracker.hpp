//
// Created by root on 1/23/17.
//

#ifndef SUPERVISORY_FRAME_CLASS_INDEPENDENT_TRACKER_HPP
#define SUPERVISORY_FRAME_CLASS_INDEPENDENT_TRACKER_HPP

#include "target.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;
/**
 * this tracker does not care about the class of the target
 */

class ClassIndependentTracker{
public:
    virtual Rect getUpdateRegion(const Mat& preImage, const Mat& curImage, const Rect& preRegion) = 0;
};
#endif //SUPERVISORY_FRAME_CLASS_INDEPENDENT_TRACKER_HPP

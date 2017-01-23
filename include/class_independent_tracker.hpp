//
// Created by root on 1/23/17.
//

#ifndef SUPERVISORY_FRAME_CLASS_INDEPENDENT_TRACKER_HPP
#define SUPERVISORY_FRAME_CLASS_INDEPENDENT_TRACKER_HPP

#include <opencv/cv.hpp>
#include "target.hpp"
using namespace cv;
using namespace std;
/**
 * this tracker does not care about the class of the target
 */

class ClassIndependentTracker{
public:
    Rect getUpdateRegion(const Mat& preImage, const Mat& curImage, const Rect& preRegion);
};
#endif //SUPERVISORY_FRAME_CLASS_INDEPENDENT_TRACKER_HPP

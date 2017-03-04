//
// Created by dujiajun on 3/4/17.
//

#ifndef SUPERVISORY_FRAME_OPENCV_UTIL_HPP
#define SUPERVISORY_FRAME_OPENCV_UTIL_HPP

#include <opencv/cv.hpp>
using namespace cv;

class OpencvUtil {
public:
    static double getOverlapRate(Rect r1, Rect r2);
};
#endif //SUPERVISORY_FRAME_OPENCV_UTIL_HPP

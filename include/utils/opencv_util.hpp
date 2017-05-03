//
// Created by dujiajun on 3/4/17.
//

#ifndef SUPERVISORY_FRAME_OPENCV_UTIL_HPP
#define SUPERVISORY_FRAME_OPENCV_UTIL_HPP

#include <opencv/cv.hpp>
#include <vector>
using namespace cv;
using namespace std;
class OpencvUtil {
public:
    static double getOverlapRate(Rect r1, Rect r2);
    static void makeRegionsVaild(vector<Rect> &regions, const Mat &image);
};
#endif //SUPERVISORY_FRAME_OPENCV_UTIL_HPP

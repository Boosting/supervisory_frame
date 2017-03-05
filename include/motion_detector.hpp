//
// Created by dujiajun on 3/5/17.
//

#ifndef SUPERVISORY_FRAME_MOTION_DETECTOR_HPP
#define SUPERVISORY_FRAME_MOTION_DETECTOR_HPP

#include <opencv/cv.hpp>
#include <vector>
using namespace cv;
using namespace std;

class MotionDetector{
public:
    /**
     * @brief Detect the moving objects from the image.
     * @param Image The image.
     * @return A vector of regions.
     */
    virtual vector<Rect> detect(const Mat &image) = 0;
};
#endif //SUPERVISORY_FRAME_MOTION_DETECTOR_HPP

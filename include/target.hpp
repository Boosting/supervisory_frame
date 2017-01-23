//
// Created by dujiajun on 1/23/17.
//

#ifndef SUPERVISORY_FRAME_TARGET_HPP
#define SUPERVISORY_FRAME_TARGET_HPP
#include <opencv/cv.hpp>
using namespace cv;
/**
 *
 */
class Target {
private:
    Rect region;
    Mat image;
public:
    Rect getRegion() const;
    void setRegion(Rect r);
    Mat getImage() const;
    void setImage(Mat i);
};

#endif //SUPERVISORY_FRAME_TARGET_HPP

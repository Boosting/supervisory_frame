//
// Created by root on 1/23/17.
//

#ifndef SUPERVISORY_FRAME_CLASS_INDEPENDENT_TRACKER_HPP
#define SUPERVISORY_FRAME_CLASS_INDEPENDENT_TRACKER_HPP

#include "target.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;

/**
 * @brief A tracker which does not care about the class of the target.
 * It doesn't depend on the target's class to update the target's region.
 */
class ClassIndependentTracker{
public:
    /**
     * @brief Get the updated region of the target.
     * @param preImage The previous image.
     * @param curImage The current image.
     * @param preRegion The target's region associated with the previous image.
     * @return The updated region.
     */
    virtual Rect getUpdateRegion(const Mat& preImage, const Mat& curImage, const Rect& preRegion) = 0;
};
#endif //SUPERVISORY_FRAME_CLASS_INDEPENDENT_TRACKER_HPP

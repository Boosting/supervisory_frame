//
// Created by root on 1/23/17.
//

#ifndef SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP
#define SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP
#include "target.hpp"
class MultiTargetDetector{
public:
    vector<Target> detectTargets(const Mat& image);
};

#endif //SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP

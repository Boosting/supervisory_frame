//
// Created by dujiajun on 3/3/17.
//

#ifndef SUPERVISORY_FRAME_DETECT_TRACK_FUSION_HPP
#define SUPERVISORY_FRAME_DETECT_TRACK_FUSION_HPP

#include "target.hpp"
#include "multi_target_detector.hpp"
#include "class_independent_tracker.hpp"
#include <vector>
using namespace std;

class DetectTrackFusion{
public:
    vector<Target> detectTrack(Mat preImage, Mat curImage, vector<Target> preTargets) = 0;
};
#endif //SUPERVISORY_FRAME_DETECT_TRACK_FUSION_HPP

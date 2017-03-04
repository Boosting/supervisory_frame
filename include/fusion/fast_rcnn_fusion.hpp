//
// Created by dujiajun on 3/3/17.
//

#ifndef SUPERVISORY_FRAME_FAST_RCNN_FUSION_HPP
#define SUPERVISORY_FRAME_FAST_RCNN_FUSION_HPP

#include "detector_only_fusion.hpp"

class FastRcnnFusion: public DetectorOnlyFusion{
public:
    FastRcnnFusion();
    vector<Target> detectTrack(Mat preImage, Mat curImage, vector<Target> preTargets);
};

#endif //SUPERVISORY_FRAME_FAST_RCNN_FUSION_HPP

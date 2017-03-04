//
// Created by dujiajun on 3/3/17.
//

#ifndef SUPERVISORY_FRAME_DETECTOR_ONLY_FUSION_HPP
#define SUPERVISORY_FRAME_DETECTOR_ONLY_FUSION_HPP

#include "detect_track_fusion.hpp"

class DetectorOnlyFusion: public DetectTrackFusion{
public:
    DetectorOnlyFusion(MultiTargetDetector &multiTargetDetector);
    vector<Target> detectTrack(Mat preImage, Mat curImage, vector<Target> preTargets);
protected:
    MultiTargetDetector &detector;
};

#endif //SUPERVISORY_FRAME_DETECTOR_ONLY_FUSION_HPP

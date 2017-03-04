//
// Created by dujiajun on 3/3/17.
//

#ifndef SUPERVISORY_FRAME_STANDARD_FUSION_HPP
#define SUPERVISORY_FRAME_STANDARD_FUSION_HPP

#include "detect_track_fusion.hpp"

class StandardFusion: public DetectTrackFusion{
public:
    StandardFusion(MultiTargetDetector &multiTargetDetector, ClassIndependentTracker &classIndependentTracker);
    vector<Target> detectTrack(Mat preImage, Mat curImage, vector<Target> preTargets);
protected:
    MultiTargetDetector &detector;
    ClassIndependentTracker &tracker;

    /**
     * @brief Perform a round of detecting from the current image.
     */
    map<unsigned long long, Target> detect(Mat curImage, vector<Target> preTargets);

    /**
     * @brief Perform a round of tracking for the detected targets.
     */
    map<unsigned long long, Target> track(Mat preImage, Mat curImage, vector<Target> preTargets);

    double getOverlapRate(Rect r1, Rect r2);
};

#endif //SUPERVISORY_FRAME_STANDARD_FUSION_HPP

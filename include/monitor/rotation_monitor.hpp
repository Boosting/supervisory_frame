//
// Created by dujiajun on 2/11/17.
//

#ifndef SUPERVISORY_FRAME_ROTATION_MONITOR_HPP
#define SUPERVISORY_FRAME_ROTATION_MONITOR_HPP

#include "real_time_monitor.hpp"

class RotationMonitor: public RealTimeMonitor{
public:
    /**
     * @brief Constructor function.
     * @param a The IP address of the video stream.
     * @param d The detector.
     * @param t The tracker.
     */
    RotationMonitor(string a, MultiTargetDetector &d, ClassIndependentTracker &t);

    /**
     * @brief Perform a round of detecting and tracking.
     */
    vector<Target> detectTrack(const Mat preImage, const Mat curImage);

protected:
    /**
     * @brief Perform a round of detecting from the current image.
     */
    map<unsigned long long, Target> detect(const Mat curImage);

    /**
     * @brief Perform a round of tracking for the detected targets.
     */
    map<unsigned long long, Target> track(const Mat curImage, const Mat preImage);

    double getOverlapRate(Rect r1, Rect r2);

};

#endif //SUPERVISORY_FRAME_ROTATION_MONITOR_HPP

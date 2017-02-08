//
// Created by root on 1/23/17.
//

#ifndef VIDEOSTREAM_REAL_TIME_MONITOR_HPP
#define VIDEOSTREAM_REAL_TIME_MONITOR_HPP

#include <opencv/cv.hpp>
#include <atomic>
#include "target.hpp"
#include "class_independent_tracker.hpp"
#include "multi_target_detector.hpp"

using namespace std;
using namespace cv;
class RealTimeMonitor {
public:
    RealTimeMonitor(string a, MultiTargetDetector d, ClassIndependentTracker t);
    bool isRunning() const;
    void run();
    void stop();
    Mat getCurrentImage();
    vector<Target> getTargets();
    void detect();
    void track();
    
private:
    VideoCapture cap;
    string address;
    atomic_bool runStatus;
    vector<Target> targets; // targets needs lock to prevent read operation when written
    MultiTargetDetector detector;
    ClassIndependentTracker tracker;

};


#endif //VIDEOSTREAM_REAL_TIME_MONITOR_HPP

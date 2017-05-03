//
// Created by dujiajun on 3/3/17.
//

#ifndef SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP
#define SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP

#include "detector/caffe_detector.hpp"
#include "motion_detector/background_substraction_motion_detector.hpp"
#include "detector/faster_rcnn_detector.hpp"

class FastRcnnDetector: public CaffeDetector {
public:
    FastRcnnDetector(const string& model_file, const string& trained_file, const vector<Target::TARGET_CLASS> &itc, int gpu_id = 1);
    vector<Target> detectTargets(const Mat &image);
protected:
    map<unsigned long long, KalmanFilter> kalman_filters;
    vector<Target> preTargets;
    BackgroundSubstractionMotionDetector motion_detector;
    vector<Rect> getKalmanProposals();
    vector<Rect> getMovingProposals(const Mat &image);
    vector<Rect> getRegionProposals(const Mat &image);
    void createRoisBlob(const vector<Rect> &regions, int sp, int ep, const string &blob_name);
};

#endif //SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP

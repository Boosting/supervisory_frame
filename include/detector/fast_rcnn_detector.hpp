//
// Created by dujiajun on 3/3/17.
//

#ifndef SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP
#define SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP

#include "detector/caffe_detector.hpp"
#include "motion_detector/background_substraction_motion_detector.hpp"
#include "detector/faster_rcnn_detector.hpp"

/**
 * This FastRcnnDetector doesn't use region proposal,
 * uses region provided instead.
 */
class FastRcnnDetector: public CaffeDetector {
public:
//    FastRcnnDetector(FasterRcnnDetector &fasterRcnnDetector, const string& model_file, const string& trained_file, int gpu_id = 1);
    FastRcnnDetector(const string& model_file, const string& trained_file, const vector<Target::TARGET_CLASS> &itc, int gpu_id = 1);
    vector<Target> detectTargets(const Mat &image);
protected:
    vector<Rect> preRegions;
    BackgroundSubstractionMotionDetector motion_detector;
//    FasterRcnnDetector &fasterRcnnDetector;
//    bool useFasterRcnn;
    vector<Rect> getRegionProposals(const Mat &image);
    Blob<float>* createRoisBlob(const vector<Rect> &regions, int sp, int ep);
};

#endif //SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP

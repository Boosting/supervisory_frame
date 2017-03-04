//
// Created by dujiajun on 3/3/17.
//

#ifndef SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP
#define SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP

#include "detector/caffe_detector.hpp"

/**
 * This FastRcnnDetector doesn't use region proposal,
 * uses region provided instead.
 */
class FastRcnnDetector: public CaffeDetector {
public:
    FastRcnnDetector(const string& model_file, const string& trained_file, bool useGPU = true);
    vector<Target> detectTargets(const Mat &image);
protected:
    Blob<float>* createRoisBlob(const vector<vector<int> > &regions);
};

#endif //SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP

//
// Created by dujiajun on 2/10/17.
//

#ifndef SUPERVISORY_FRAME_FASTER_RCNN_DETECTOR_HPP
#define SUPERVISORY_FRAME_FASTER_RCNN_DETECTOR_HPP

#include "detector/caffe_detector.hpp"

class FasterRcnnDetector: public CaffeDetector {
public:
    FasterRcnnDetector(const string& model_file, const string& trained_file, bool useGPU = true);
    vector<Target> detectTargets(const Mat& image);
protected:
    Blob<float>* createImInfoBlob(const Mat& image);
};

#endif //SUPERVISORY_FRAME_FASTER_RCNN_DETECTOR_HPP

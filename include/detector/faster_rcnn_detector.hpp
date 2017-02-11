//
// Created by dujiajun on 2/10/17.
//

#ifndef SUPERVISORY_FRAME_FASTER_RCNN_DETECTOR_HPP
#define SUPERVISORY_FRAME_FASTER_RCNN_DETECTOR_HPP

#include "multi_target_detector.hpp"
#include "target.hpp"
#include <caffe/caffe.hpp>
#include <vector>
#include<iostream>
using namespace caffe;
using namespace std;

class FasterRcnnDetector: public MultiTargetDetector {
public:
    FasterRcnnDetector(const string& model_file, const string& trained_file, bool useGPU = true);
    vector<Target> detectTargets(const Mat& image);
protected:
    Blob<float>* createImageBlob(const Mat& image);
    Blob<float>* createImInfoBlob(const Mat& image);
    vector<vector<float> > getOutputData(string blob_name);
    boost::shared_ptr<Net<float> > net;
};

#endif //SUPERVISORY_FRAME_FASTER_RCNN_DETECTOR_HPP

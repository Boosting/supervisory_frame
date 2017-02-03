//
// Created by root on 1/23/17.
//

#ifndef SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP
#define SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP
#include "target.hpp"
#include <caffe/caffe.hpp>
using namespace caffe;

class MultiTargetDetector{
public:
    MultiTargetDetector(const string& model_file, const string& trained_file, bool useGPU = true);
    vector<Target> detectTargets(const Mat& image);

private:
    shared_ptr<Net<float> > caffe_net;
    //vector<Target> label_vec;
};

#endif //SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP

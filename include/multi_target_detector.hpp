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
    int roi_num = 4096;
    int cls_num = 4; // include __background__
    //vector<Target> label_vec;

    vector<vector<float> > getOutputData(shared_ptr< Net<float> > & net, string blob_name);
    vector<vector<int> > bbox_transform(const vector<vector<float> > &rois, const vector<vector<float> > &bbox_pred);
    vector<vector<int> > nms(const vector<vector<int> > &bbox, const vector<vector<float> > &cls_prob, float thresh = 0.3);
};

#endif //SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP

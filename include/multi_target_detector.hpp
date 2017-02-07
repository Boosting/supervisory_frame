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
    boost::shared_ptr<Net<float> > net;
    int roi_num = 0;
    int cls_num = 4; // include __background__
    //vector<Target> label_vec;
    Blob<float>* createImageBlob(const Mat& image);
    Blob<float>* createImInfoBlob(const Mat& image);
    vector<vector<float> > getOutputData(string blob_name);
    vector<vector<vector<int> > > bbox_transform(const vector<vector<float> > &rois, const vector<vector<float> > &bbox_pred);
    vector<vector<int> > nms(const vector<vector<vector<int> > > &bbox, const vector<vector<float> > &cls_prob, float thresh = 0.3, float min_trust_score = 0.1);
};

#endif //SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP

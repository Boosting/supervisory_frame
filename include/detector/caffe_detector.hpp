//
// Created by dujiajun on 3/3/17.
//

#ifndef SUPERVISORY_FRAME_CAFFE_DETECTOR_HPP
#define SUPERVISORY_FRAME_CAFFE_DETECTOR_HPP

#include "multi_target_detector.hpp"
#include "target.hpp"
#include <caffe/caffe.hpp>
#include <vector>
using namespace caffe;
using namespace std;

class CaffeDetector: public MultiTargetDetector {
public:
    CaffeDetector(const string& model_file, const string& trained_file, bool useGPU = true);
protected:
    Blob<float>* createImageBlob(const Mat& image);
    vector<vector<float> > getOutputData(string blob_name);
    boost::shared_ptr<Net<float> > net;
};

#endif //SUPERVISORY_FRAME_CAFFE_DETECTOR_HPP

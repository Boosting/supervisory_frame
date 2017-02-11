//
// Created by root on 1/23/17.
//

#ifndef SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP
#define SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP
#include "target.hpp"
#include <vector>
using namespace std;

class MultiTargetDetector{
public:
    MultiTargetDetector();
    virtual vector<Target> detectTargets(const Mat& image) = 0;
protected:
    int roi_num = 0;
    int cls_num = 4; // include __background__
    vector<Target::TARGET_CLASS> idToClass;
    vector<vector<vector<int> > > bbox_transform(const vector<vector<float> > &rois, const vector<vector<float> > &bbox_pred);
    vector<vector<int> > nms(const vector<vector<vector<int> > > &bbox, const vector<vector<float> > &cls_prob, float thresh = 0.7, float min_trust_score = 0.1);
};

#endif //SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP

//
// Created by dujiajun on 2/10/17.
//

#ifndef SUPERVISORY_FRAME_FASTER_RCNN_DETECTOR_HPP
#define SUPERVISORY_FRAME_FASTER_RCNN_DETECTOR_HPP

#include "detector/caffe_detector.hpp"

class FasterRcnnDetector: public CaffeDetector {
public:
    FasterRcnnDetector(const string& model_file, const string& trained_file, const vector<Target::TARGET_CLASS> &itc, int gpu_id = 2);
    vector<Target> detectTargets(const Mat& image);
protected:
    void createImInfoBlob(const Mat& image, const string &blob_name);
    vector<Rect> getRegions(vector<vector<float> > &rois);
};

#endif //SUPERVISORY_FRAME_FASTER_RCNN_DETECTOR_HPP

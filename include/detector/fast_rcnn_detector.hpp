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
    /**
     * @brief x1, y1, x2 ,y2 between 0~1
     * @param image
     * @return
     */
    vector<vector<float> > getMovingRegions(const Mat &image);
    Blob<float>* createRoisBlob(const vector<vector<float> > &regions);
    Ptr<BackgroundSubtractorMOG2> mog;
};

#endif //SUPERVISORY_FRAME_FAST_RCNN_DETECTOR_HPP

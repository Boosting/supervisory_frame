//
// Created by dujiajun on 2/12/17.
//

#ifndef SUPERVISORY_FRAME_YOLO_DETECTOR_HPP
#define SUPERVISORY_FRAME_YOLO_DETECTOR_HPP

#include "multi_target_detector.hpp"

#undef __cplusplus
extern "C" {
#include "image.h"
#include "network.h"
}
#define __cplusplus 201103L //C++ 11 use 201103L

class YoloDetector: public MultiTargetDetector {
public:
    YoloDetector(bool useGPU = true);
    vector<Target> detectTargets(const Mat& image);
protected:
    network darknet_network;
    image createImage(const Mat& image);
    vector<vector<float> > get_detections(const image &im, int num, float thresh, box *boxes, float **probs, int classes);
    vector<vector<float> > kitti_detect(const image &im, const network &net);
};
#endif //SUPERVISORY_FRAME_YOLO_DETECTOR_HPP

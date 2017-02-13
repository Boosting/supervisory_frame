//
// Created by dujiajun on 2/12/17.
//

#ifndef SUPERVISORY_FRAME_YOLO_DETECTOR_HPP
#define SUPERVISORY_FRAME_YOLO_DETECTOR_HPP

#include "multi_target_detector.hpp"
#include "image.h"
#include "network.h"

class YoloDetector: public MultiTargetDetector {
public:
    YoloDetector(bool useGPU = true);
    vector<Target> detectTargets(const Mat& image);
protected:
    network darknet_network;
    image createImage(const Mat& image);
};
#endif //SUPERVISORY_FRAME_YOLO_DETECTOR_HPP

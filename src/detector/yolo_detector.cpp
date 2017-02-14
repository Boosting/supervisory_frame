//
// Created by dujiajun on 2/13/17.
//

#include "detector/yolo_detector.hpp"
#include "kitti_detection.h"

YoloDetector::YoloDetector(bool useGPU) {
    darknet_network = get_network();
}

vector<Target> YoloDetector::detectTargets(const Mat& image) {
    image im = createImage(image);
    vector<vector<int> > bbox_cls = kitti_detect(im, darknet_network);
    vector<Target> targets = bboxToTarget(bbox_cls);
    return targets;
}

image YoloDetector::createImage(const Mat& image) {
    image im;
    int height = image.rows, width = image.cols, channels = 3;
    im.h = height, im.w = width, im.c = channels;
    im.data = new float[height * width * channels];
    for(int j=0;j<height;j++)
    {
        const uchar *mat_data = image.ptr<uchar>(j);
        for(int i=0;i<width;i++){
            for(int k=0;k<channels;k++){
                int dst_index = i + width * (j + height * k);
                im.data[dst_index] = (float)(*mat_data) / 255.0;
                mat_data++;
            }
        }
    }
    return im;
}
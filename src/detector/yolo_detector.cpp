//
// Created by dujiajun on 2/13/17.
//
#include "detector/yolo_detector.hpp"

#undef __cplusplus
extern "C" {
#include "region_layer.h"
#include "parser.h"
#include "box.h"
#include "utils.h"
#include "cuda_renamed.h"
}
#define __cplusplus 201103L

YoloDetector::YoloDetector(bool useGPU) {
    char cfgfile[100] = "/home/dujiajun/darknet/cfg/yolo-kitti.cfg";
    char weightfile[100] = "/home/dujiajun/darknet/yolo-kitti_final.weights";
#ifdef GPU
    gpu_index = 0;
#else
	gpu_index = -1;
#endif
    darknet_network = parse_network_cfg(cfgfile);
    load_weights(&darknet_network, weightfile);
	set_batch_network(&darknet_network, 1);
}

vector<Target> YoloDetector::detectTargets(const Mat& mat_image) {
    image im = createImage(mat_image);
    vector<vector<float> > bbox_cls_score = kitti_detect(im, darknet_network);
    vector<Target> targets = bboxToTarget(bbox_cls_score);
    return targets;
}

image YoloDetector::createImage(const Mat& mat_image) {
    image im;
    int height = mat_image.rows, width = mat_image.cols, channels = 3;
    im.h = height, im.w = width, im.c = channels;
    im.data = new float[height * width * channels];
    for(int j=0;j<height;j++)
    {
        const uchar *mat_data = mat_image.ptr<uchar>(j);
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

vector<vector<float> > YoloDetector::get_detections(const image &im, int num, float thresh, box *boxes, float **probs, int classes) {
    vector<vector<float> > bbox_cls_score(0, vector<float>(6));
    for(int i = 0; i < num; ++i){
        int class_id = max_index(probs[i], classes);
        float prob = probs[i][class_id];
        if(prob < thresh) continue;

        box b = boxes[i];
        int x1  = (b.x-b.w/2.)*im.w;
        int x2 = (b.x+b.w/2.)*im.w;
        int y1   = (b.y-b.h/2.)*im.h;
        int y2   = (b.y+b.h/2.)*im.h;
        if(x1 < 0) x1 = 0;
        if(x2 > im.w-1) x2 = im.w-1;
        if(y1 < 0) y1 = 0;
        if(y2 > im.h-1) y2 = im.h-1;
        vector<float> item(6);
        item[0]=x1, item[1]=y1, item[2]=x2, item[3]=y2, item[4]=class_id+1, item[5]=prob;
        bbox_cls_score.push_back(item);
    }
    return bbox_cls_score;
}

vector<vector<float> > YoloDetector::kitti_detect(const image &im, const network &net){
    float thresh = 0.24, nms = 0.4, hier_thresh = 0.5;
    int width = net.w, height = net.h;
    layer l = net.layers[net.n-1];
	int output_num = l.w*l.h*l.n;
    image image_resized = resize_image(im, net.w, net.h);
    box *boxes = new box[output_num];
    float **probs = new float*[output_num];
    for(int j = 0; j < output_num; ++j) {
        probs[j] = new float[l.classes + 1];
    }
    network_predict(net, image_resized.data);
	get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
 	if (l.softmax_tree && nms) do_nms_obj(boxes, probs, output_num, l.classes, nms);
    else if (nms) do_nms_sort(boxes, probs, output_num, l.classes, nms);

    vector<vector<float> > bbox_cls_score= get_detections(im, output_num, thresh, boxes, probs, l.classes);

    delete [] boxes;
    for(int j = 0; j < output_num; ++j) {
        delete [] probs[j];
    }
    delete [] probs;
    return bbox_cls_score;
}

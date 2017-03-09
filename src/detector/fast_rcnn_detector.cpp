//
// Created by dujiajun on 3/3/17.
//

#include "detector/fast_rcnn_detector.hpp"

FastRcnnDetector::FastRcnnDetector(const string& model_file, const string& trained_file, bool useGPU)
        :CaffeDetector(model_file, trained_file, useGPU) {
    // VOC 1+20 classes
    idToClass = {
            Target::UNKNOWN,
            Target::UNKNOWN, Target::BICYCLE, Target::UNKNOWN, Target::UNKNOWN,
            Target::UNKNOWN, Target::BUS, Target::CAR, Target::UNKNOWN,
            Target::UNKNOWN, Target::UNKNOWN, Target::UNKNOWN, Target::UNKNOWN,
            Target::UNKNOWN, Target::MOTORBIKE, Target::PERSON, Target::UNKNOWN,
            Target::UNKNOWN, Target::UNKNOWN, Target::TRAIN, Target::UNKNOWN
    };
}

vector<Target> FastRcnnDetector::detectTargets(const Mat &image) {
    Blob<float>* image_blob = createImageBlob(image);
    vector<Rect> regions = getRegionProposals(image);
    if(regions.empty()) return vector<Target>();
    int region_num = regions.size();
    Blob<float>* rois = createRoisBlob(regions);
    vector<Blob<float>* > bottom = {image_blob, rois};
    float type = 0.0;
    vector<int> image_shape = {1, 3, image.rows, image.cols};
    vector<int> rois_shape = {region_num, 5};
    net->input_blobs()[0]->Reshape(image_shape);
    net->input_blobs()[1]->Reshape(rois_shape);
    net->Reshape();

    clock_t time1, time2;
    time1 = clock();
    net->Forward(bottom, &type);
    time2 = clock();
    cout<<"forward use time: "<< (double)(time2 - time1) / CLOCKS_PER_SEC <<endl;

    vector<vector<float> > cls_prob = getOutputData("cls_prob");
    vector<vector<float> > bbox_pred = getOutputData("bbox_pred");
    vector<vector<Rect> > bbox = bbox_transform(regions, bbox_pred, image);
    vector<Target> targets = nms(bbox, cls_prob, 0.7, 0.01);
    return targets;
}

vector<Rect> FastRcnnDetector::getRegionProposals(const Mat &image) {
    vector<Rect> regions = motion_detector.detect(image);
    return regions;
}

Blob<float>* FastRcnnDetector::createRoisBlob(const vector<Rect> &regions){
    int region_num = regions.size();
    vector<int> rois_shape={region_num, 5};
    Blob<float>* rois_blob = new Blob<float>(rois_shape); //may cause memory leak
    float* rois_blob_cpu_data = rois_blob->mutable_cpu_data();
    for(int i=0;i<regions.size();i++){
        int index = 5*i;
        const Rect &region = regions[i];
        int x1 = region.x, y1 = region.y;
        int w = region.width, h = region.height;
        int x2 = x1 + w - 1, y2 = y1 + h - 1;
        rois_blob_cpu_data[index] = 0;
        rois_blob_cpu_data[index+1] = x1;
        rois_blob_cpu_data[index+2] = y1;
        rois_blob_cpu_data[index+3] = x2;
        rois_blob_cpu_data[index+4] = y2;
    }
    return rois_blob;
}

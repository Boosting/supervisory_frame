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
    vector<Rect> rect_regions = motion_detector.detect(image);
    if(rect_regions.empty()) return vector<Target>();

    int region_num = rect_regions.size();
    vector<vector<float> > regions(region_num);
    for(int i=0;i<region_num;i++){
        float x1=rect_regions[i].x, y1=rect_regions[i].y;
        float w=rect_regions[i].width, h=rect_regions[i].height;
        regions[i] = {x1, y1, x1+w, y1+h};
    }
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
    cout<<"forward use time: "<< (double)(time2 - time1) / CLOCKS_PER_SEC <<endl<<endl<<endl;

    vector<vector<float> > cls_prob = getOutputData("cls_prob");
    vector<vector<float> > bbox_pred = getOutputData("bbox_pred");
    vector<vector<vector<float> > > bbox = bbox_transform(regions, bbox_pred, image);

    vector<vector<float> > bbox_cls_score = nms(bbox, cls_prob); //bbox + cls = 4 + 1
    vector<Target> target_vec = bboxToTarget(bbox_cls_score, idToClass);
    return target_vec;
}

Blob<float>* FastRcnnDetector::createRoisBlob(const vector<vector<float> > &regions){
    int region_num = regions.size();
    vector<int> rois_shape={region_num, 5};
    Blob<float>* rois_blob = new Blob<float>(rois_shape); //may cause memory leak
    float* rois_blob_cpu_data = rois_blob->mutable_cpu_data();
    for(int i=0;i<regions.size();i++){
        int index = 5*i;
        rois_blob_cpu_data[index] = 0;
        rois_blob_cpu_data[index+1] = regions[i][0];
        rois_blob_cpu_data[index+2] = regions[i][1];
        rois_blob_cpu_data[index+3] = regions[i][2];
        rois_blob_cpu_data[index+4] = regions[i][3];
    }
    return rois_blob;
}
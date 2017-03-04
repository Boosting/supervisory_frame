//
// Created by dujiajun on 2/10/17.
//
#include "detector/faster_rcnn_detector.hpp"
FasterRcnnDetector::FasterRcnnDetector(const string& model_file, const string& trained_file, bool useGPU)
        :CaffeDetector(model_file, trained_file, useGPU) {}

vector<Target> FasterRcnnDetector::detectTargets(const Mat& image) {
    //Only single-image batch implemented, and no image pyramid

    Blob<float>* image_blob = createImageBlob(image);
    Blob<float>* im_info_blob = createImInfoBlob(image);
    vector<Blob<float>* > bottom = {image_blob, im_info_blob};
    float type = 0.0;
    vector<int> image_shape = {1, 3, image.rows, image.cols};
    net->input_blobs()[0]->Reshape(image_shape);
    net->Reshape();

    clock_t time1, time2;
    time1 = clock();
    net->Forward(bottom, &type);
    time2 = clock();
    cout<<"forward use time: "<< (double)(time2 - time1) / CLOCKS_PER_SEC <<endl<<endl<<endl;

    vector<vector<float> > rois = getOutputData("rois");
    vector<vector<float> > cls_prob = getOutputData("cls_prob");
    vector<vector<float> > bbox_pred = getOutputData("bbox_pred");
//printVec(rois);
//printVec(cls_prob);
//printVec(bbox_pred);
    vector<vector<vector<int> > > bbox = this->bbox_transform(rois, bbox_pred);

    vector<vector<int> > bbox_cls = nms(bbox, cls_prob); //bbox + cls = 4 + 1
    vector<Target> target_vec = bboxToTarget(bbox_cls);
    return target_vec;
}

Blob<float>* FasterRcnnDetector::createImInfoBlob(const Mat& image){
    int image_height = image.rows, image_width = image.cols;
    vector<int> im_info_shape={1,3};
    Blob<float>* im_info_blob = new Blob<float>(im_info_shape);
    float* data = im_info_blob->mutable_cpu_data();
    data[0]=image_height, data[1]=image_width, data[2]=1;
    return im_info_blob;
}


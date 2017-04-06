//
// Created by dujiajun on 2/10/17.
//
#include "detector/faster_rcnn_detector.hpp"
FasterRcnnDetector::FasterRcnnDetector(const string& model_file, const string& trained_file, const vector<Target::TARGET_CLASS> &itc, int gpu_id)
        :CaffeDetector(model_file, trained_file, itc, gpu_id) {}

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
    vector<Rect> regions = getRegions(rois);
    vector<vector<float> > cls_prob = getOutputData("cls_prob");
    vector<vector<float> > bbox_pred = getOutputData("bbox_pred");

    vector<vector<Rect> > bbox = bbox_transform(regions, bbox_pred, image);
    vector<Target> targets = nms(bbox, cls_prob, 0.5, 0.1);
    return targets;
}

Blob<float>* FasterRcnnDetector::createImInfoBlob(const Mat& image){
    int image_height = image.rows, image_width = image.cols;
    vector<int> im_info_shape={1,3};
    Blob<float>* im_info_blob = new Blob<float>(im_info_shape);
    float* data = im_info_blob->mutable_cpu_data();
    data[0]=image_height, data[1]=image_width, data[2]=1;
    return im_info_blob;
}

vector<Rect> FasterRcnnDetector::getRegions(vector<vector<float> > &rois){
    vector<Rect> regions(rois.size());
    for(int i=0;i<rois.size();i++){
        int x1 = rois[i][1], y1 = rois[i][2], x2 = rois[i][3], y2 = rois[i][4];
        regions[i] = {x1, y1, x2-x1+1, y2-y1+1};
    }
    return regions;
}


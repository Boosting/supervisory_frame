//
// Created by dujiajun on 3/3/17.
//

#include "detector/fast_rcnn_detector.hpp"

FastRcnnDetector::FastRcnnDetector(const string& model_file, const string& trained_file, bool useGPU)
        :CaffeDetector(model_file, trained_file, useGPU) {}

vector<Target> FastRcnnDetector::detectTargets(const Mat &image) {
    Blob<float>* image_blob = createImageBlob(image);
    vector<vector<float> > regions = getMovingRegions(image);
    Blob<float>* rois = createRoisBlob(regions);
    vector<Blob<float>* > bottom = {image_blob, rois};
    float type = 0.0;
    vector<int> image_shape = {1, 3, image.rows, image.cols};
    net->input_blobs()[0]->Reshape(image_shape);
    net->Reshape();

    clock_t time1, time2;
    time1 = clock();
    net->Forward(bottom, &type);
    time2 = clock();
    cout<<"forward use time: "<< (double)(time2 - time1) / CLOCKS_PER_SEC <<endl<<endl<<endl;

    vector<vector<float> > cls_prob = getOutputData("cls_prob");
    vector<vector<float> > bbox_pred = getOutputData("bbox_pred");
    vector<vector<vector<float> > > bbox = bbox_transform(regions, bbox_pred);

    vector<vector<float> > bbox_cls_score = nms(bbox, cls_prob); //bbox + cls = 4 + 1
    vector<Target> target_vec = bboxToTarget(bbox_cls_score);
    return target_vec;
}

vector<vector<float> > FastRcnnDetector::getMovingRegions(const Mat &image) {
    Mat GaussianImage, foreground;
    GaussianBlur(image, GaussianImage, {11, 11}, 2);
    mog->apply(GaussianImage, foreground, 0.001);
    erode(foreground, foreground, cv::Mat());
    dilate(foreground, foreground, cv::Mat());
}

Blob<float>* createRoisBlob(const vector<vector<float> > &regions){
    int region_num = regions.size();
    vector<int> rois_shape={region_num, 5};
    Blob<float>* rois_blob = new Blob<float>(rois_shape); //may cause memory leak
    float* rois_blob_cpu_data = rois_blob->mutable_cpu_data();
    for(int i=0;i<regions.size();i++){
        int index = 5*i;
        rois_blob_cpu_data[index] = i;
        rois_blob_cpu_data[index+1] = regions[i][0];
        rois_blob_cpu_data[index+2] = regions[i][1];
        rois_blob_cpu_data[index+3] = regions[i][2];
        rois_blob_cpu_data[index+4] = regions[i][3];
    }
    return rois_blob;
}
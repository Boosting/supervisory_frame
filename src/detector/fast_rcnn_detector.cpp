//
// Created by dujiajun on 3/3/17.
//

#include "detector/fast_rcnn_detector.hpp"

FastRcnnDetector::FastRcnnDetector(const string& model_file, const string& trained_file, bool useGPU)
        :CaffeDetector(model_file, trained_file, useGPU) {}

vector<Target> FastRcnnDetector::detectTargets(const Mat &image) {
    Blob<float>* image_blob = createImageBlob(image);
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
    vector<vector<vector<int> > > bbox = bbox_transform(regions, bbox_pred);

    vector<vector<int> > bbox_cls = nms(bbox, cls_prob); //bbox + cls = 4 + 1
    vector<Target> target_vec = bboxToTarget(bbox_cls);
    return target_vec;
}

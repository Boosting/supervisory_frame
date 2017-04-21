//
// Created by dujiajun on 2/10/17.
//
#include "detector/faster_rcnn_detector.hpp"
FasterRcnnDetector::FasterRcnnDetector(const string& model_file, const string& trained_file, const vector<Target::TARGET_CLASS> &itc, int gpu_id)
        :CaffeDetector(model_file, trained_file, itc, gpu_id) {}

vector<Target> FasterRcnnDetector::detectTargets(const Mat& image) {
    createImInfoBlob(image, "im_info")
    vector<int> image_shape = {1, 3, image.rows, image.cols};
    net->input_blobs()[0]->Reshape(image_shape);
    net->Reshape();

    clock_t time1, time2;
    time1 = clock();
    net->ForwardPrefilled();
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

void FasterRcnnDetector::createImInfoBlob(const Mat& image, const string &blob_name){
    boost::shared_ptr<Blob<float> > blob_ptr = net->blob_by_name(blob_name);
    float* blob_data = blob_ptr->mutable_cpu_data();
    int image_height = image.rows, image_width = image.cols;
    blob_data[0] = image_height, blob_data[1] = image_width, blob_data[2] = 1;
}

vector<Rect> FasterRcnnDetector::getRegions(vector<vector<float> > &rois){
    vector<Rect> regions(rois.size());
    for(int i=0;i<rois.size();i++){
        int x1 = rois[i][1], y1 = rois[i][2], x2 = rois[i][3], y2 = rois[i][4];
        regions[i] = {x1, y1, x2-x1+1, y2-y1+1};
    }
    return regions;
}


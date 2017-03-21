//
// Created by dujiajun on 3/3/17.
//

#include "detector/fast_rcnn_detector.hpp"
#include "utils/selective_search.hpp"

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
    vector<Rect> regions = getRegionProposals(image);
    if(regions.empty()) return vector<Target>();
    int region_num = regions.size();
    int batch_size = 4;

    Blob<float>* image_blob = createImageBlob(image);
    float type = 0.0;
    net->input_blobs()[0]->Reshape({1, 3, image.rows, image.cols});
    net->input_blobs()[1]->Reshape({batch_size, 5});
    net->Reshape();

    vector<vector<float> > cls_prob(region_num);
    vector<vector<float> > bbox_pred(region_num);
    clock_t time1, time2;
    time1 = clock();
    for(int i=0;i<regions.size();i+=batch_size) {
        int sp = i, ep;
        if(sp+batch_size>regions.size()){
            ep = regions.size()-1;
            net->input_blobs()[1]->Reshape({ep - sp + 1, 5});
            net->Reshape();
        } else{
            ep = sp+batch_size-1;
        }
        Blob<float>* rois = createRoisBlob(regions, sp, ep);
        vector<Blob<float>* > bottom = {image_blob, rois};
        net->Forward(bottom, &type);
        vector<vector<float> > part_cls_prob = getOutputData("cls_prob");
        vector<vector<float> > part_bbox_pred = getOutputData("bbox_pred");
        for(int j=sp;j<=ep;j++){
            cls_prob[j] = part_cls_prob[j-sp];
            bbox_pred[j] = part_bbox_pred[j-sp];
        }
    }
    time2 = clock();
    cout<<"forward use time: "<< (double)(time2 - time1) / CLOCKS_PER_SEC <<endl;

    vector<vector<Rect> > bbox = bbox_transform(regions, bbox_pred, image);
    vector<Target> targets = nms(bbox, cls_prob, 0.5, 0.1);

    preRegions = vector<Rect>(targets.size());
    for(int i=0;i<targets.size();i++){
        preRegions[i] = targets[i].getRegion();
    }
    return targets;
}

vector<Rect> FastRcnnDetector::getRegionProposals(const Mat &image) {
    int image_width = image.cols, image_height = image.rows;
    vector<Rect> region_proposals = preRegions;
    vector<Rect> moving_regions = motion_detector.detect(image);
    for(Rect &r1: moving_regions){
        Mat partImage;
        int x1 = r1.x, y1 = r1.y;
        int w1 = r1.width, h1 = r1.height;
        int center_x = x1 + w1 / 2, center_y = y1 + h1 / 2;
        getRectSubPix(image, {w1, h1}, {center_x, center_y}, partImage);
        vector<Rect> regions = region_proposal::selectiveSearch(partImage, 500, 0.8, 50, 1000, 100000, 2.5 );
        for(Rect &r2: regions){
            int x2 = min(x1 + r2.x, image_width-1), y2 = min(y1 + r2.y, image_height-1);
            int w2 = min(r2.width, image_width - x2), h2 = min(r2.height, image_height - y2);
            region_proposals.push_back({x2, y2, w2, h2});
        }
    }
    return region_proposals;
}

Blob<float>* FastRcnnDetector::createRoisBlob(const vector<Rect> &regions, int sp, int ep){
    vector<int> rois_shape={ep - sp + 1, 5};
    Blob<float>* rois_blob = new Blob<float>(rois_shape); //may cause memory leak
    float* rois_blob_cpu_data = rois_blob->mutable_cpu_data();
    for(int i=sp;i<=ep;i++){
        int index = 5 * (i - sp);
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

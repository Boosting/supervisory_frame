//
// Created by dujiajun on 3/3/17.
//

#include <utils/opencv_util.hpp>
#include "detector/fast_rcnn_detector.hpp"
#include "utils/selective_search.hpp"

FastRcnnDetector::FastRcnnDetector(const string& model_file, const string& trained_file, const vector<Target::TARGET_CLASS> &itc, int gpu_id)
        :CaffeDetector(model_file, trained_file, itc, gpu_id){
}

vector<Target> FastRcnnDetector::detectTargets(const Mat &image) {
    vector<Rect> regions = getRegionProposals(image);
    if(regions.empty()) return vector<Target>();
    int region_num = regions.size();
    int batch_size = 128;

    net->input_blobs()[0]->Reshape({1, 3, image.rows, image.cols});
    net->input_blobs()[1]->Reshape({batch_size, 5});
    net->Reshape();
    createImageBlob(image, "data");
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
        createRoisBlob(regions, sp, ep, "rois");
		net->ForwardPrefilled();
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
    preTargets = targets;
    return targets;
}

vector<Rect> FastRcnnDetector::getKalmanProposals() {
    vector<Rect> kalman_proposals;
    for(auto p: kalman_filters){
        unsigned long long id = p.first;
        KalmanFilter &kalman_filter = p.second;
        Mat prediction = kalman_filter.predict();
        float center_x = prediction.at<float>(0), center_y = prediction.at<float>(1);
        for(Target &target: preTargets){
            if(target.getId()==id) {
                Rect region = target.getRegion();
                int width = region.width, height = region.height;
                int x = center_x - width / 2.0, y = center_y - height / 2.0;
                kalman_proposals.push_back({x, y, width, height});
                break;
            }
        }
    }
    return kalman_proposals;
}
vector<Rect> FastRcnnDetector::getMovingProposals(const Mat &image){
    int image_width = image.cols, image_height = image.rows;
    vector<Rect> moving_proposals;
    vector<Rect> moving_regions = motion_detector.detect(image);
    for(Rect &r1: moving_regions){
        Mat partImage;
        int x1 = r1.x, y1 = r1.y;
        int w1 = r1.width, h1 = r1.height;
        int center_x = x1 + w1 / 2, center_y = y1 + h1 / 2;
        getRectSubPix(image, {w1, h1}, {center_x, center_y}, partImage);
        vector<Rect> regions = region_proposal::selectiveSearch(partImage, 500, 0.8, 50, 1000, 100000, 2.5 );
        for(Rect &r2: regions){
            int x2 = x1 + r2.x, y2 = y1 + r2.y;
            int w2 = r2.width, h2 = r2.height;
            moving_proposals.push_back({x2, y2, w2, h2});
        }
    }
    return moving_proposals;
}
vector<Rect> FastRcnnDetector::getRegionProposals(const Mat &image) {
    vector<Rect> kalman_proposals = getKalmanProposals();
    vector<Rect> moving_proposals = getMovingProposals(image);
    kalman_proposals.insert(kalman_proposals.end(), moving_proposals.begin(), moving_proposals.end());
    OpencvUtil::makeRegionsVaild(kalman_proposals, image);
    return kalman_proposals;
}

void FastRcnnDetector::createRoisBlob(const vector<Rect> &regions, int sp, int ep, const string &blob_name){
    boost::shared_ptr<Blob<float> > blob_ptr = net->blob_by_name(blob_name);
    float* blob_data = blob_ptr->mutable_cpu_data();
    for(int i=sp;i<=ep;i++){
        int index = 5 * (i - sp);
        const Rect &region = regions[i];
        int x1 = region.x, y1 = region.y;
        int w = region.width, h = region.height;
        int x2 = x1 + w - 1, y2 = y1 + h - 1;
        blob_data[index] = 0;
        blob_data[index+1] = x1;
        blob_data[index+2] = y1;
        blob_data[index+3] = x2;
        blob_data[index+4] = y2;
    }
}

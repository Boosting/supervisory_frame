//
// Created by dujiajun on 2/2/17.
//
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "multi_target_detector.hpp"
using namespace caffe;
using namespace std;

MultiTargetDetector::MultiTargetDetector() {
    idToClass = {Target::UNKNOWN, Target::CAR, Target::PEDESTRIAN, Target::CYCLIST};
    cls_num = idToClass.size();
}

vector<vector<vector<int> > > MultiTargetDetector::bbox_transform(const vector<vector<float> > &rois, const vector<vector<float> > &bbox_pred){
    int roi_num = rois.size();
    vector<vector<vector<int> > > bbox(roi_num, vector<vector<int> >(cls_num, vector<int>(4)));
    for(int i=0;i<roi_num;i++) {
        float x1 = rois[i][1], y1 = rois[i][2], x2 = rois[i][3], y2 = rois[i][4]; //rois[i][0] is not position
        float width = x2 - x1 + 1, height = y2 - y1 + 1, center_x = x1 + width * 0.5, center_y = y1 + height * 0.5;
        for (int j = 0; j < cls_num; j++) {
            int offset = j * cls_num;
            float dx = bbox_pred[i][offset], dy = bbox_pred[i][offset+1], dw = bbox_pred[i][offset+2], dh = bbox_pred[i][offset+3];
            float pred_width = width * exp(dw), pred_height = height * exp(dh);
            float pred_center_x = dx * width + center_x, pred_center_y = dy * height + center_y;
            float pred_x1 = pred_center_x - pred_width * 0.5, pred_x2 = pred_center_x + pred_width * 0.5;
            float pred_y1 = pred_center_y - pred_height * 0.5, pred_y2 = pred_center_y + pred_height * 0.5;
            bbox[i][j][0] = pred_x1, bbox[i][j][1] = pred_y1, bbox[i][j][2] = pred_x2, bbox[i][j][3] = pred_y2;  // convert float to int may be ambiguous
        }
    }
    return bbox;
}

vector<vector<int> > MultiTargetDetector::nms(const vector<vector<vector<int> > > &bbox, const vector<vector<float> > &cls_prob, float thresh, float min_trust_score) {
    vector<vector<int> > bbox_cls; //x1, y1, x2, y2, cls
    int roi_num = bbox.size();
    for(int cls_id=1;cls_id<cls_num;cls_id++){
        vector<vector<float> > bbox_score;
        for(int i=0;i<roi_num;i++){
            // can speed up by delete low score bbox
            float score=cls_prob[i][cls_id];
            int x1=bbox[i][cls_id][0], y1=bbox[i][cls_id][1], x2=bbox[i][cls_id][2], y2=bbox[i][cls_id][3];
            if(score<min_trust_score || x1>x2 || y1>y2) continue; // remove low score or wrong position
            bbox_score.push_back({x1, y1, x2, y2, score});
        }
        sort(bbox_score.begin(), bbox_score.end(),
             [](const vector<float> &bbox1, const vector<float> &bbox2) -> bool {
                 return bbox1[4]>bbox2[4];
             }
        );

        cout<<"cls id: "<<cls_id<<" bbox_score num: "<<bbox_score.size()<<endl;

        vector<bool> is_suppressed(bbox_score.size(), false);
        for(int i=0;i<bbox_score.size();i++){
            if(is_suppressed[i]) continue;
            float lx1=bbox_score[i][0], ly1=bbox_score[i][1], lx2=bbox_score[i][2], ly2=bbox_score[i][3];
            for(int j=i+1;j<bbox_score.size();j++){
                float sx1=bbox_score[j][0], sy1=bbox_score[j][1], sx2=bbox_score[j][2], sy2=bbox_score[j][3];
                float x1max=max(lx1,sx1), x2min=min(lx2,sx2), y1max=max(ly1,sy1), y2min=min(ly2,sy2);
                float overlapWidth = x2min - x1max + 1;
                float overlapHeight = y2min - y1max + 1;
                float small_bbox_size = (sx2-sx1+1)*(sy2-sy1+1);
                if(overlapHeight > 0 && overlapWidth > 0) {
                    float overlapRate = (overlapWidth * overlapHeight) / small_bbox_size; //avoid divide 0 in the code before
                    if (overlapRate > thresh) {
                        is_suppressed[j] = true;
                    }
                }
            }
        }
        for(int i=0;i<bbox_score.size();i++){
            if(!is_suppressed[i]){
                int x1=bbox_score[i][0], y1=bbox_score[i][1], x2=bbox_score[i][2], y2=bbox_score[i][3];
                bbox_cls.push_back({x1, y1, x2, y2, cls_id});
            }
        }
    }
    return bbox_cls;
}

//int main()
//{
//    string model_file="/home/dujiajun/py-faster-rcnn/models/kitti/VGG16/faster_rcnn_end2end/test.prototxt";
//    string trained_file="/home/dujiajun/py-faster-rcnn/data/kitti/VGG16/faster_rcnn_end2end.caffemodel";
//    string image_file="/home/dujiajun/kitti/testing/image_2/000456.png";
//    MultiTargetDetector detector(model_file, trained_file);
//    Mat image;
//    image=cv::imread(image_file);
//    cout<<"height: "<<image.rows<<" width: "<<image.cols<<endl;
//    detector.detectTargets(image);
//    return 0;
//}

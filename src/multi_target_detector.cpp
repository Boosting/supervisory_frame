//
// Created by dujiajun on 2/2/17.
//
#include <opencv2/opencv.hpp>
#include <memory>
#include "multi_target_detector.hpp"
using namespace std;

vector<vector<Rect> > MultiTargetDetector::bbox_transform(const vector<vector<float> > &rois, const vector<vector<float> > &bbox_pred, const Mat& image){
    if(rois.empty()) return vector<vector<Rect> >();
	int image_height = image.rows, image_width = image.cols;
	int cls_num = bbox_pred[0].size();
    int roi_num = rois.size();
    vector<vector<Rect> > bbox(roi_num, vector<Rect>(cls_num));
    for(int i=0;i<roi_num;i++) {
        float x1 = rois[i][0], y1 = rois[i][1], x2 = rois[i][2], y2 = rois[i][3];
        float width = x2 - x1 + 1, height = y2 - y1 + 1, center_x = x1 + width * 0.5, center_y = y1 + height * 0.5;
        for (int j = 0; j < cls_num; j++) {
            int offset = j * cls_num;
            float dx = bbox_pred[i][offset], dy = bbox_pred[i][offset+1], dw = bbox_pred[i][offset+2], dh = bbox_pred[i][offset+3];
            float pred_width = width * exp(dw), pred_height = height * exp(dh);
            float pred_center_x = dx * width + center_x, pred_center_y = dy * height + center_y;
            int pred_x1 = pred_center_x - pred_width * 0.5, pred_x2 = pred_center_x + pred_width * 0.5;
            int pred_y1 = pred_center_y - pred_height * 0.5, pred_y2 = pred_center_y + pred_height * 0.5;
            pred_x1 = max(pred_x1, 0), pred_y1 = max(pred_y1, 0);
            pred_x2 = min(pred_x2, image_width-1), pred_y2 = min(pred_y2, image_height-1);
            int pred_w = max(pred_x2 - pred_x1 + 1, 1);
            int pred_h = max(pred_y2 - pred_y1 + 1, 1);
            bbox[i][j] = {pred_x1, pred_y1, pred_w, pred_h};
        }
    }
    return bbox;
}

vector<Target> MultiTargetDetector::nms(const vector<vector<Rect> > &bbox, const vector<vector<float> > &cls_prob, float thresh, float min_trust_score) {
    vector<Target> targets;
    if(bbox.empty()) return vector<Target>();
	
	int cls_num = cls_prob[0].size();
    int roi_num = bbox.size();
    for(int cls_id=1;cls_id<cls_num;cls_id++){
        vector<vector<float> > bbox_score;
        for(int i=0;i<roi_num;i++){
            // can speed up by delete low score bbox
            float score=cls_prob[i][cls_id];
            int x1=bbox[i][cls_id].x, y1=bbox[i][cls_id].y;
            int w=bbox[i][cls_id].width, h=bbox[i][cls_id].height;
            int x2=x1+w-1, y2=y1+h-1;
            if(score<min_trust_score) continue; // remove low score
            bbox_score.push_back({x1, y1, x2, y2, score});
        }
        sort(bbox_score.begin(), bbox_score.end(),
             [](const vector<float> &bbox1, const vector<float> &bbox2) -> bool {
                 return bbox1[4]>bbox2[4];
             }
        );

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
                float score = bbox_score[i][4];
                cout<<"x1: "<<x1<<" y1: "<<y1<<" x2: "<<x2<<" y2: "<<y2<<endl;
                cout<<"cls: "<<cls_id<<endl;
                cout<<"score: "<<score<<endl;
                Target::TARGET_CLASS target_class = (cls_id >= 0 && cls_id < idToClass.size()) ? idToClass[cls_id] : Target::UNKNOWN;
                target.setClass(target_class);
                target.setRegion({x1, y1, x2-x1+1, y2-y1+1});
                target.setScore(score);
                targets.push_back(target);
            }
        }
    }
    return targets;
}

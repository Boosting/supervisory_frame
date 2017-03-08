//
// Created by dujiajun on 2/2/17.
//
#include <opencv2/opencv.hpp>
#include <memory>
#include "multi_target_detector.hpp"
using namespace std;

vector<vector<vector<float> > > MultiTargetDetector::bbox_transform(const vector<vector<float> > &rois, const vector<vector<float> > &bbox_pred, const Mat& image){
    if(rois.empty()) return vector<vector<vector<float> > >();
	int image_height = image.rows, image_width = image.cols;
	int cls_num = bbox_pred[0].size();
    int roi_num = rois.size();
    vector<vector<vector<float> > > bbox(roi_num, vector<vector<float> >(cls_num, vector<float>(4)));
    for(int i=0;i<roi_num;i++) {
        float x1 = rois[i][0], y1 = rois[i][1], x2 = rois[i][2], y2 = rois[i][3];
        float width = x2 - x1 + 1, height = y2 - y1 + 1, center_x = x1 + width * 0.5, center_y = y1 + height * 0.5;
        for (int j = 0; j < cls_num; j++) {
            int offset = j * cls_num;
            float dx = bbox_pred[i][offset], dy = bbox_pred[i][offset+1], dw = bbox_pred[i][offset+2], dh = bbox_pred[i][offset+3];
            float pred_width = width * exp(dw), pred_height = height * exp(dh);
            float pred_center_x = dx * width + center_x, pred_center_y = dy * height + center_y;
            float pred_x1 = pred_center_x - pred_width * 0.5, pred_x2 = pred_center_x + pred_width * 0.5;
            float pred_y1 = pred_center_y - pred_height * 0.5, pred_y2 = pred_center_y + pred_height * 0.5;
            bbox[i][j][0] = max(int(pred_x1), 0), bbox[i][j][1] = max(int(pred_y1), 0);
            bbox[i][j][2] = min(int(pred_x2), image_height-1), bbox[i][j][3] = min(int(pred_y2), image_width-1);  // convert float to int may be ambiguous
        }
    }
    return bbox;
}

vector<vector<float> > MultiTargetDetector::nms(const vector<vector<vector<float> > > &bbox, const vector<vector<float> > &cls_prob, float thresh, float min_trust_score) {
    vector<vector<float> > bbox_cls_score; //x1, y1, x2, y2, cls, score
    if(bbox.empty()) return vector<vector<float> >();
	
	int cls_num = cls_prob[0].size();
    int roi_num = bbox.size();
    for(int cls_id=1;cls_id<cls_num;cls_id++){
        vector<vector<float> > bbox_score;
        for(int i=0;i<roi_num;i++){
            // can speed up by delete low score bbox
            float score=cls_prob[i][cls_id];
            int x1=bbox[i][cls_id][0], y1=bbox[i][cls_id][1], x2=bbox[i][cls_id][2], y2=bbox[i][cls_id][3];
            if(score<min_trust_score || x1>=x2 || y1>=y2) continue; // remove low score or wrong position
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
                bbox_cls_score.push_back({x1, y1, x2, y2, cls_id, score});
            }
        }
    }
    return bbox_cls_score;
}


vector<Target> MultiTargetDetector::bboxToTarget(vector<vector<float> > &bbox_cls_score, vector<Target::TARGET_CLASS> &idToClass) {
    for(int i=0;i<bbox_cls_score.size();i++){
        cout<<"x1: "<<bbox_cls_score[i][0]<<" y1: "<<bbox_cls_score[i][1]<<" x2: "<<bbox_cls_score[i][2]<<" y2: "<<bbox_cls_score[i][3]<<endl;
        cout<<"cls: "<<bbox_cls_score[i][4]<<endl;
        cout<<"score: "<<bbox_cls_score[i][5]<<endl;
    }

    //translate cls to Target
    int target_num = bbox_cls_score.size();
    vector<Target> target_vec(target_num);
    for(int i=0;i<target_num;i++){
        Target &target = target_vec[i];
        vector<float> &vec=bbox_cls_score[i];
        int class_id = vec[4];
        Target::TARGET_CLASS target_class = (class_id >= 0 && class_id < idToClass.size()) ? idToClass[class_id] : Target::UNKNOWN;
        target.setClass(target_class);
        int x1=vec[0], y1=vec[1], x2=vec[2], y2=vec[3];
        target.setRegion({x1, y1, x2-x1+1, y2-y1+1});
        target.setScore(vec[5]);
    }
    return target_vec;
}

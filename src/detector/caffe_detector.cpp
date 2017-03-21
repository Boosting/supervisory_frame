//
// Created by dujiajun on 3/3/17.
//

#include "detector/caffe_detector.hpp"

CaffeDetector::CaffeDetector(const string& model_file, const string& trained_file, int gpu_id):MultiTargetDetector() {
    if (gpu_id>=0) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_id); //may implement detecting gpu id automatically later
    } else Caffe::set_mode(Caffe::CPU);

    /* Load the network. */
    net.reset(new Net<float>(model_file, caffe::TEST));
    net->CopyTrainedLayersFrom(trained_file);
}

Blob<float>* CaffeDetector::createImageBlob(const Mat& image){
    int image_num = 1, image_channels = 3, image_height = image.rows, image_width = image.cols;
    vector<int> image_shape={image_num, image_channels, image_height, image_width};
    Blob<float>* image_blob = new Blob<float>(image_shape); //may cause memory leak
    float* image_blob_data = image_blob->mutable_cpu_data();

    for(int j=0;j<image_height;j++) //may need speed up
    {
        const uchar *data = image.ptr<uchar>(j);
        for(int k=0;k<image_width;k++){
            for(int i=0;i<image_channels;i++){
                int pos=(i*image_height+j)*image_width+k;
                image_blob_data[pos] = (int)(*data);
                data++;
            }
        }
    }
    return image_blob;
}

vector<vector<float> > CaffeDetector::getOutputData(string blob_name)
{
    boost::shared_ptr<Blob<float> > blob_ptr = net->blob_by_name(blob_name);
    const float* blob_data = blob_ptr->cpu_data();
    int num = blob_ptr->num();
    int channels = blob_ptr->channels();
    vector<vector<float> > output_data(num, vector<float>(channels));
    for(int i=0;i<num;i++){
        for(int j=0;j<channels;j++){
            output_data[i][j] = blob_data[i*channels+j];
        }
    }
    return output_data;
}

vector<vector<Rect> > CaffeDetector::bbox_transform(const vector<Rect> &rois, const vector<vector<float> > &bbox_pred, const Mat& image){
    if(rois.empty()) return vector<vector<Rect> >();
    int image_height = image.rows, image_width = image.cols;
    int cls_num = bbox_pred[0].size() / 4; // each rois predict cls_num * 4 bbox, (dx, dy, dw, dh)
    int roi_num = rois.size();
    vector<vector<Rect> > bbox(roi_num, vector<Rect>(cls_num));
    for(int i=0;i<roi_num;i++) {
        float x1 = rois[i].x, y1 = rois[i].y;
        float width = rois[i].width, height = rois[i].height;
        float x2 = x1 + width - 1, y2 = y1 + height - 1;
        float center_x = x1 + width * 0.5, center_y = y1 + height * 0.5;
        for (int j = 0; j < cls_num; j++) {
            float dx = bbox_pred[i][j*4], dy = bbox_pred[i][j*4+1], dw = bbox_pred[i][j*4+2], dh = bbox_pred[i][j*4+3];
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

vector<Target> CaffeDetector::nms(const vector<vector<Rect> > &bbox, const vector<vector<float> > &cls_prob, float thresh, float min_trust_score) {
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
                Target target;
                int x1=bbox_score[i][0], y1=bbox_score[i][1], x2=bbox_score[i][2], y2=bbox_score[i][3];
                float score = bbox_score[i][4];
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
//
// Created by dujiajun on 2/2/17.
//
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "multi_target_detector.hpp"
using namespace caffe;
using namespace std;

MultiTargetDetector::MultiTargetDetector(const string& model_file, const string& trained_file, bool useGPU) {
    Caffe::set_phase(Caffe::TEST);
    if(useGPU) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(0); //may implement detecting gpu id automatically later
    }
    else Caffe::set_mode(Caffe::CPU);

    /* Load the network. */
    net.reset(new Net<float>(model_file, Caffe::TEST));
    net->CopyTrainedLayersFrom(trained_file);

//    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
//    CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net->input_blobs()[0];
    int num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";

    //Blob<float>* output_layer = net->output_blobs()[0];
    //CHECK_EQ(labels_.size(), output_layer->channels())
    //    << "Number of labels is different from the output layer dimension.";

}
Blob<float>* MultiTargetDetector::createImageBlob(const Mat& image){
    int image_channels = 3, image_height = image.cols, image_width = image.rows;
    Blob<float>* image_blob = new Blob<float>(1, image_channels, image_height, image_width); //may cause memory leak

    //get the blobproto
    BlobProto blob_proto;
    blob_proto.set_num(1);
    blob_proto.set_channels(image_channels);
    blob_proto.set_height(image_height);
    blob_proto.set_width(image_width);
    const int data_size = image_channels*image_height*image_width;
    for (int i = 0; i < data_size; ++i) {
        blob_proto.add_data(0.);
    }

    for(int j=0;j<image_height;j++)
    {
        const uchar *data = image.ptr<uchar>(j);
        for(int k=0;k<image_width;k++){
            for(int i=0;i<image_channels;i++){
                int pos=(i*image_height+j)*image_width+k;
                blob_proto.set_data(pos, blob_proto.data(pos) + (uint8_t)(*data));
                data++;
            }
        }
    }

    //set data into blob
    image_blob->FromProto(blob_proto);

    return image_blob;
}
vector<Target> MultiTargetDetector::detectTargets(const Mat& image) {
    //Only single-image batch implemented, and no image pyramid

    Blob<float>* image_blob = createImageBlob(image);
    vector<Blob<float>*> bottom; bottom.push_back(image_blob);
    float type = 0.0;
    net->Forward(bottom, &type);

    vector<vector<float> > rois = getOutputData("rois");
    vector<vector<float> > cls_prob = getOutputData("cls_prob");
    vector<vector<float> > bbox_pred = getOutputData("bbox_pred");

    vector<vector<int> > bbox = bbox_transform(rois, bbox_pred);

    vector<vector<int> > bbox_cls = nms(bbox, cls_prob); //bbox + cls = 4 + 1
    for(int i=0;i<bbox_cls.size();i++){
        cout<<"x1: "<<bbox_cls[i][0]<<" y1: "<<bbox_cls[i][1]<<" x2: "<<bbox_cls[i][2]<<" y2: "<<bbox_cls[i][3]<<endl;
        cout<<"cls: "<<bbox_cls[i][4]<<endl;
    }
    //translate cls to Target
    return vector<Target>();
}

vector<vector<float> > MultiTargetDetector::getOutputData(string blob_name)
{
	boost::shared_ptr<Blob<float> > blob_ptr = net->blob_by_name(blob_name);
    int blob_cnt = blob_ptr->count();
    const float* blob_data = blob_ptr->cpu_data();
    int second_layer_size = blob_cnt / roi_num;
    vector<vector<float> > output_data(roi_num, vector<float>(second_layer_size));
    for(int i=0;i<roi_num;i++){
        for(int j=0;j<second_layer_size;j++){
            output_data[i][j] = blob_data[i*roi_num+j];
        }
    }
    return output_data;
}

vector<vector<int> > MultiTargetDetector::bbox_transform(const vector<vector<float> > &rois, const vector<vector<float> > &bbox_pred){
    vector<vector<int> > bbox(roi_num, vector<int>(4));
    for(int i=0;i<roi_num;i++){
        float x1=rois[i][1], y1=rois[i][2], x2=rois[i][3],y2=rois[i][4]; //rois[i][0] is not position
        float width=x2-x1+1, height=y2-y1+1, center_x=x1+width*0.5, center_y=y1+height*0.5;
        float dx=bbox_pred[i][0], dy=bbox_pred[i][1], dw=bbox_pred[i][2], dh=bbox_pred[i][3];
        float pred_width = width * exp(dw), pred_height = height * exp(dh);
        float pred_center_x = dx * width + center_x, pred_center_y = dy * height + center_y;
        float pred_x1 = pred_center_x - pred_width * 0.5, pred_x2 = pred_center_x + pred_width * 0.5;
        float pred_y1 = pred_center_y - pred_height * 0.5, pred_y2 = pred_center_y + pred_height * 0.5;
        bbox[i][0]=pred_x1, bbox[i][1]=pred_y1, bbox[i][2]=pred_x2, bbox[i][3]=pred_y2;  // convert float to int may be ambiguous
    }
    return bbox;
}

vector<vector<int> > MultiTargetDetector::nms(const vector<vector<int> > &bbox, const vector<vector<float> > &cls_prob, float thresh) {
    vector<vector<int> > bbox_cls; //x1, y1, x2, y2, cls
    for(int cls_id=1;cls_id<cls_num;cls_id++){
        vector<vector<float> > bbox_score;
        for(int i=0;i<roi_num;i++){
            // can speed up by delete low score bbox
            float score=cls_prob[i][cls_id];
            int x1=bbox[i][0], y1=bbox[i][1], x2=bbox[i][2], y2=bbox[i][3];
            if(x1>x2||y1>y2) continue; // delete wrong position
            bbox_score.push_back({x1, y1, x2, y2, score});
        }
        sort(bbox_score.begin(), bbox_score.end(),
             [](const vector<float> &bbox1, const vector<float> &bbox2) -> bool {
                 return bbox1[4]>bbox2[4];
             }
        );

        vector<bool> is_suppressed(bbox_score.size(), false);
        for(int i=0;i<bbox_score.size()-1;i++){
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

int main()
{
    string model_file="/home/dujiajun/py-faster-rcnn/models/kitti/VGG16/faster_rcnn_end2end/test.prototxt";
    string trained_file="/home/dujiajun/py-faster-rcnn/data/kitti/VGG16/faster_rcnn_end2end.caffemodel";
    string image_file="/home/dujiajun/kitti/testing/image_2/000456.png";
    MultiTargetDetector detector(model_file, trained_file);
    Mat image;
    image=cv::imread(image_file);
    cout<<"height: "<<image.rows<<" width: "<<image.cols<<endl;
    detector.detectTargets(image);
    return 0;
}

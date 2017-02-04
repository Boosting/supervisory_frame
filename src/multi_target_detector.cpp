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

MultiTargetDetector::MultiTargetDetector(const string& model_file, const string& trained_file, bool useGPU = true) {
    if(useGPU) Caffe::set_mode(Caffe::GPU);
    else Caffe::set_mode(Caffe::CPU);

    /* Load the network. */
    net.reset(new Net<float>(model_file, TEST));
    net->CopyTrainedLayersFrom(trained_file);

//    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
//    CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";

    //Blob<float>* output_layer = net->output_blobs()[0];
    //CHECK_EQ(labels_.size(), output_layer->channels())
    //    << "Number of labels is different from the output layer dimension.";

}

vector<Target> MultiTargetDetector::detectTargets(const Mat& image) {
    //Only single-image batch implemented, and no image pyramid

    Blob<float>* input_layer = net->input_blobs()[0];
    int image_height = image.cols, image_width = image.rows;
    input_layer->Reshape(1, 3, image_height, image_width); // channel: 3
    net->Reshape();

    vector<cv::Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }

    // need to preprocess the image
    cv::split(image, *input_channels);
    net->ForwardPrefilled();

    vector<vector<float> > rois = getOutputData(net, "rois");
    vector<vector<float> > cls_prob = getOutputData(net, "cls_prob");
    vector<vector<float> > bbox_pred = getOutputData(net, "bbox_pred");

    vector<vector<float> > bbox = bbox_transform(rois, bbox_pred, cls_prob);

    vector<int> bbox_cls = nms(bbox, cls_prob); //bbox + cls = 4 + 1
    //translate cls to Target
    return ;
}

vector<vector<float> > MultiTargetDetector::getOutputData(shared_ptr< Net<float> > & net, string blob_name)
{
    shared_ptr<Blob<float> > blob_ptr = net->blob_by_name(blob_name);
    int blob_cnt = blob_ptr->count();
    float* blob_data = blob_ptr->cpu_data();
    int second_layer_size = blob_cnt / roi_num;
    vector<vector<float> > output_data(roi_num, second_layer_size);
    for(int i=0;i<roi_num;i++){
        for(int j=0;j<second_layer_size;j++){
            output_data[i][j] = blob_data[i*roi_num+j];
        }
    }
    return output_data;
}

vector<vector<float> > MultiTargetDetector::bbox_transform(const vector<vector<float> > &rois, const vector<vector<float> > &bbox_pred){
    vector<vector<float> > bbox(roi_num, vector<float>(4));
    for(int i=0;i<roi_num;i++){
        float x1=rois[i][1], y1=rois[i][2], x2=rois[i][3],y2=rois[i][4]; //rois[i][0] is not position
        float width=x2-x1+1, height=y2-y1+1, center_x=x1+width*0.5, center_y=y1+height*0.5;
        float dx=bbox_pred[i][0], dy=bbox_pred[i][1], dw=bbox_pred[i][2], dh=bbox_pred[i][3];
        float pred_width = width * exp(dw), pred_height = height * exp(dh);
        float pred_center_x = dx * width + center_x, pred_center_y = dy * height + center_y;
        float pred_x1 = pred_center_x - pred_width * 0.5, pred_x2 = pred_center_x + pred_width * 0.5;
        float pred_y1 = pred_center_y - pred_height * 0.5, pred_y2 = pred_center_y + pred_height * 0.5;
        bbox[i][0]=pred_x1, bbox[i][1]=pred_y1, bbox[i][2]=pred_x2, bbox[i][3]=pred_y2;
    }
    return bbox;
}
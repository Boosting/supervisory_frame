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
    caffe_net.reset(new Net<float>(model_file, TEST));
    caffe_net->CopyTrainedLayersFrom(trained_file);

//    CHECK_EQ(caffe_net->num_inputs(), 1) << "Network should have exactly one input.";
//    CHECK_EQ(caffe_net->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = caffe_net->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";

    //Blob<float>* output_layer = caffe_net->output_blobs()[0];
    //CHECK_EQ(labels_.size(), output_layer->channels())
    //    << "Number of labels is different from the output layer dimension.";

}

vector<float> MultiTargetDetector::detectTargets(const Mat& image) {
    Blob<float>* input_layer = caffe_net->input_blobs()[0];
    int image_height = image.cols, image_width = image.rows;
    input_layer->Reshape(1, 3, image_height, image_width); // channel: 3
    caffe_net->Reshape();

    vector<cv::Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }

    // need to preprocess the image
    cv::split(image, *input_channels);
    caffe_net->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = caffe_net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + 4 + 1;
    return vector<float>(begin, end);
}
//
// Created by dujiajun on 3/3/17.
//

#include "detector/caffe_detector.hpp"

CaffeDetector::CaffeDetector(const string& model_file, const string& trained_file, bool useGPU):MultiTargetDetector() {
    if (useGPU) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(2); //may implement detecting gpu id automatically later
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

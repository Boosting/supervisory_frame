//
// Created by dujiajun on 2/10/17.
//
#include "detector/faster_rcnn_detector.hpp"
FasterRcnnDetector::FasterRcnnDetector(const string& model_file, const string& trained_file, bool useGPU):MultiTargetDetector() {
    if (useGPU) {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(0); //may implement detecting gpu id automatically later
    } else Caffe::set_mode(Caffe::CPU);

    /* Load the network. */
    net.reset(new Net<float>(model_file, caffe::TEST));
    net->CopyTrainedLayersFrom(trained_file);

    Blob<float> *input_layer = net->input_blobs()[0];
    int num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
}

vector<Target> FasterRcnnDetector::detectTargets(const Mat& image) {
    //Only single-image batch implemented, and no image pyramid

    Blob<float>* image_blob = createImageBlob(image);
    Blob<float>* im_info_blob = createImInfoBlob(image);
    vector<Blob<float>*> bottom = {image_blob, im_info_blob};
    float type = 0.0;
    vector<int> image_shape = {1, 3, image.rows, image.cols};
    net->input_blobs()[0]->Reshape(image_shape);
    net->Reshape();

    clock_t time1, time2;
    time1 = clock();
    net->Forward(bottom, &type);
    time2 = clock();
    cout<<"forward use time: "<< (double)(time2 - time1) / CLOCKS_PER_SEC <<endl<<endl<<endl;

    vector<vector<float> > rois = getOutputData("rois");
    vector<vector<float> > cls_prob = getOutputData("cls_prob");
    vector<vector<float> > bbox_pred = getOutputData("bbox_pred");
    roi_num = rois.size();
//printVec(rois);
//printVec(cls_prob);
//printVec(bbox_pred);
    vector<vector<vector<int> > > bbox = this->bbox_transform(rois, bbox_pred);

    vector<vector<int> > bbox_cls = nms(bbox, cls_prob); //bbox + cls = 4 + 1
    for(int i=0;i<bbox_cls.size();i++){
        cout<<"x1: "<<bbox_cls[i][0]<<" y1: "<<bbox_cls[i][1]<<" x2: "<<bbox_cls[i][2]<<" y2: "<<bbox_cls[i][3]<<endl;
        cout<<"cls: "<<bbox_cls[i][4]<<endl;
    }

    //translate cls to Target
    int target_num = bbox_cls.size();
    vector<Target> target_vec(target_num);
    for(int i=0;i<target_num;i++){
        Target &target = target_vec[i];
        vector<int> &vec=bbox_cls[i];
        int class_id = vec[4];
        Target::TARGET_CLASS target_class = (class_id >= 0 && class_id < 4) ? idToClass[class_id] : Target::UNKNOWN;
        target.setClass(target_class);
        int x1=vec[0], y1=vec[1], x2=vec[2], y2=vec[3];
        target.setRegion({x1, y1, x2-x1+1, y2-y1+1});
    }

    return target_vec;
}

Blob<float>* FasterRcnnDetector::createImageBlob(const Mat& image){
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

Blob<float>* FasterRcnnDetector::createImInfoBlob(const Mat& image){
    int image_height = image.rows, image_width = image.cols;
    vector<int> im_info_shape={1,3};
    Blob<float>* im_info_blob = new Blob<float>(im_info_shape);
    float* data = im_info_blob->mutable_cpu_data();
    data[0]=image_height, data[1]=image_width, data[2]=1;
    return im_info_blob;
}

vector<vector<float> > FasterRcnnDetector::getOutputData(string blob_name)
{
    boost::shared_ptr<Blob<float> > blob_ptr = net->blob_by_name(blob_name);
    const float* blob_data = blob_ptr->cpu_data();
    int num = blob_ptr->num();
    int channels = blob_ptr->channels();
    vector<vector<float> > output_data(num, vector<float>(channels));
    cout<<"num "<<num<<" channels "<<channels<<" "<<endl;
    for(int i=0;i<num;i++){
        for(int j=0;j<channels;j++){
            output_data[i][j] = blob_data[i*channels+j];
        }
    }
    return output_data;
}
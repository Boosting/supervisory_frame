//
// Created by dujiajun on 5/2/17.
//
#include "detector/faster_rcnn_detector.hpp"
#include "fusion/detector_only_fusion.hpp"
#include "real_time_monitor.hpp"

int main(){
    int gpu_id=0;
    string video_address;
    cout<<"please input gpu id:"<<endl;
    cin>>gpu_id;
    cout<<"please input video address"<<endl;
    cin>>video_address;
    vector<Target::TARGET_CLASS> voc_itc = {
            Target::UNKNOWN,
            Target::UNKNOWN, Target::BICYCLE, Target::UNKNOWN, Target::UNKNOWN,
            Target::UNKNOWN, Target::BUS, Target::CAR, Target::UNKNOWN,
            Target::UNKNOWN, Target::UNKNOWN, Target::UNKNOWN, Target::UNKNOWN,
            Target::UNKNOWN, Target::MOTORBIKE, Target::PERSON, Target::UNKNOWN,
            Target::UNKNOWN, Target::UNKNOWN, Target::TRAIN, Target::UNKNOWN
    };
    vector<Target::TARGET_CLASS> kitti_itc = {
            Target::UNKNOWN, Target::CAR, Target::PERSON, Target::BICYCLE
    };
    string voc_prototxt = "/home/dujiajun/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt";
    string voc_model = "/home/dujiajun/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel";
    string kitti_prototxt = "/home/dujiajun/py-faster-rcnn/models/kitti/VGG16/faster_rcnn_end2end/test.prototxt";
    string kitti_model = "/home/dujiajun/py-faster-rcnn/output/faster_rcnn_end2end/kitti_2012_train/vgg16_faster_rcnn_iter_70000.caffemodel";
    FasterRcnnDetector detector(kitti_prototxt, kitti_model, voc_itc, gpu_id);
    DetectorOnlyFusion fusion(detector);
    Displayer displayer;
    RealTimeMonitor monitor(video_address, fusion, displayer);
    monitor.run();
    while(monitor.isRunning()){
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}


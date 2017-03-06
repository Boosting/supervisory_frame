//
// Created by dujiajun on 3/3/17.
//

#include "detector/fast_rcnn_detector.hpp"
#include "fusion/detector_only_fusion.hpp"
#include "real_time_monitor.hpp"

int main(){
    string address="/home/dujiajun/car_person_video.mp4";
    FastRcnnDetector detector("/home/dujiajun/fast-rcnn/models/VGG16/test.prototxt", "/home/dujiajun/fast-rcnn/data/fast_rcnn_models/vgg16_fast_rcnn_iter_40000.caffemodel", false);
    DetectorOnlyFusion fusion(detector);
    Displayer displayer;
    RealTimeMonitor monitor(address, fusion, displayer);
    monitor.run();
    while(monitor.isRunning()){
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}
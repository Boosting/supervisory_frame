//
// Created by dujiajun on 3/3/17.
//

#include "detector/fast_rcnn_detector.hpp"
#include "fusion/detector_only_fusion.hpp"
#include "real_time_monitor.hpp"

int main(){
    string address="/home/dujiajun/CUHKSquare.mpg";
    string prototxt = "/home/dujiajun/fast-rcnn/models/VGG_CNN_M_1024/test.prototxt";
    string trained_model = "/home/dujiajun/fast-rcnn/data/fast_rcnn_models/vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel";
    FastRcnnDetector detector(prototxt, trained_model);
    DetectorOnlyFusion fusion(detector);
    Displayer displayer;
    RealTimeMonitor monitor(address, fusion, displayer);
    monitor.run();
    while(monitor.isRunning()){
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}
//
// Created by dujiajun on 3/3/17.
//

#include "detector/faster_rcnn_detector.hpp"
#include "fusion/detector_only_fusion.hpp"
#include "real_time_monitor.hpp"

int main(){
    string address="/home/dujiajun/CUHKSquare.mpg";
    string prototxt = "/home/dujiajun/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt";
    string trained_model = "/home/dujiajun/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel";
    FasterRcnnDetector detector(prototxt, trained_model);
    DetectorOnlyFusion fusion(detector);
    Displayer displayer;
    RealTimeMonitor monitor(address, fusion, displayer);
    monitor.run();
    while(monitor.isRunning()){
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

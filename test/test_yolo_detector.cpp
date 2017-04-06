//
// Created by dujiajun on 3/4/17.
//

#include "detector/yolo_detector.hpp"
#include "fusion/detector_only_fusion.hpp"
#include "real_time_monitor.hpp"

int main(){
    coco_itc = {
            Target::PERSON,     Target::BICYCLE,        Target::CAR,            Target::MOTORBIKE,
            Target::UNKNOWN,    Target::BUS,            Target::TRAIN,          Target::TRUCK,
            Target::UNKNOWN,    Target::TRAFFIC_LIGHT,  Target::FIRE_HYDRANT,   Target::STOP_SIGN
    }; // MSCOCO 80 classes
    for(int i=12;i<80;i++) coco_itc.push_back(Target::UNKNOWN);

//    string address="rtsp://192.168.1.201:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream";
    string address = "/home/dujiajun/CUHKSquare.mpg";
    YoloDetector detector(coco_itc);
    DetectorOnlyFusion fusion(detector);
    Displayer displayer;
    RealTimeMonitor monitor(address, fusion, displayer);
    monitor.run();
    while(monitor.isRunning()){
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

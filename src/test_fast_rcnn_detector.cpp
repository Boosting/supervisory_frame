//
// Created by dujiajun on 3/3/17.
//

#include "detector/fast_rcnn_detector.hpp"
#include "tracker/kcf_tracker.hpp"
#include "fusion/standard_fusion.hpp"
#include "real_time_monitor.hpp"

int main(){
    string address="/home/dujiajun/car_person_video.mp4";
    FastRcnnDetector detector;
    KcfTracker tracker;
    StandardFusion fusion(detector, tracker);
    Displayer displayer;
    RealTimeMonitor monitor(address, fusion, displayer);
    monitor.run();
    while(monitor.isRunning()){
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}
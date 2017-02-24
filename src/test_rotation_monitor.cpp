//
// Created by dujiajun on 2/11/17.
//


#include "detector/yolo_detector.hpp"
#include "tracker/kcf_tracker.hpp"
#include "monitor/rotation_monitor.hpp"

int main(){
    string address="/home/dujiajun/car_person_video.mp4";
    YoloDetector detector;
    KcfTracker tracker;
    Displayer displayer;
    RotationMonitor monitor(address, detector, tracker, displayer);
    monitor.run();
    while(monitor.isRunning()){
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

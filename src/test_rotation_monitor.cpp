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
    RotationMonitor monitor(address, detector, tracker);
    monitor.run();
    while(monitor.isRunning()){
        Mat image = monitor.getCurrentImage();
        vector<Target> targets = monitor.getTargets();
        if(image.empty()) continue;
        for(Target &target: targets){
            Rect region = target.getRegion();
            rectangle(image, region, Scalar( 255, 0, 0 ), 1, 1);
        }
        imshow("monitor", image);
        waitKey(10);
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

//
// Created by dujiajun on 2/11/17.
//


#include "detector/faster_rcnn_detector.hpp"
#include "tracker/kcf_tracker.hpp"
#include "monitor/rotation_monitor.hpp"

int main(){
    string address="/home/dujiajun/car_person_video.mp4";
    string model_file="/home/dujiajun/py-faster-rcnn/models/kitti/VGG16/faster_rcnn_end2end/test.prototxt";
    string trained_file="/home/dujiajun/py-faster-rcnn/data/kitti/VGG16/faster_rcnn_end2end.caffemodel";
    FasterRcnnDetector detector(model_file, trained_file);
    KcfTracker tracker;
    RotationMonitor monitor(address, detector, tracker);
    monitor.run();
    while (monitor.isRunning())
    {
        Mat frame = monitor.getCurrentImage();
        if(frame.empty()) continue;
        vector<Target> targets = monitor.getTargets();
        for(Target target: targets){
            Rect region = target.getRegion();
            rectangle(frame, region, Scalar( 255, 0, 0 ), 1, 1);
        }
        imshow("monitor", frame);
        this_thread::sleep_for(chrono::milliseconds(20));
    }
}
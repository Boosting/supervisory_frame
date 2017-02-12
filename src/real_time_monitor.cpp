//
// Created by dujiajun on 1/13/17.
//
#include "real_time_monitor.hpp"
#include "detector/faster_rcnn_detector.hpp"
#include "tracker/kcf_tracker.hpp"
#include<iostream>
using namespace std;

RealTimeMonitor::RealTimeMonitor(string a, MultiTargetDetector &d, ClassIndependentTracker &t)
        :address(a), detector(d), tracker(t), runStatus(false), stopSignal(false) {}

void RealTimeMonitor::loop(){
    int cnt=0;
    while(!stopSignal) {
        if(cnt==0) detect();
        else track();
        this_thread::sleep_for(chrono::milliseconds(10));
        cnt++;
        if(cnt==20) cnt=0;
    }
    runStatus = false;
    stopSignal = false;
}
bool RealTimeMonitor::isRunning() const {
    return runStatus;
}
void RealTimeMonitor::run(){ // two threads call run() at the same time may cause error
    if(runStatus) return;
    runStatus = true; stopSignal = false;
    if(!cap.isOpened()) cap.open(address);
    thread loopThread(loop, this);
    loopThread.detach();
}

void RealTimeMonitor::stop(){
    if(!runStatus) return;
    stopSignal = true;
}

Mat RealTimeMonitor::getCurrentImage(){
    boost::shared_lock<boost::shared_mutex> lock(image_mutex);
    return currentImage;
}

Mat RealTimeMonitor::getUpdatedImage() {
    boost::unique_lock<boost::shared_mutex> lock(image_mutex);
    if(cap.isOpened()) {
        cap >> currentImage;
    }
    return currentImage;
}
vector<Target> RealTimeMonitor::getTargets(){
    return targets;
}
void RealTimeMonitor::detect(){
    Mat curImage = getUpdatedImage();
    if(!curImage.empty()) {
        targets = detector.detectTargets(curImage);
        for (Target t: targets) {
            t.setImage(curImage);
        }
    }
}
void RealTimeMonitor::track(){
    for(Target &t: targets){
        Mat preImage = t.getImage(), curImage = getUpdatedImage();
        if(curImage.empty()) continue;
        Rect preRegion = t.getRegion();
        t.setImage(curImage);
        t.setRegion(tracker.getUpdateRegion(preImage, curImage, preRegion));
    }
}

int main(){
    string address="/home/dujiajun/car_person_video.mp4";
    string model_file="/home/dujiajun/py-faster-rcnn/models/kitti/VGG16/faster_rcnn_end2end/test.prototxt";
    string trained_file="/home/dujiajun/py-faster-rcnn/data/kitti/VGG16/faster_rcnn_end2end.caffemodel";
    FasterRcnnDetector detector(model_file, trained_file);
    KcfTracker tracker;
    RealTimeMonitor monitor(address, detector, tracker);
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

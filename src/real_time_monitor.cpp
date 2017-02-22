//
// Created by dujiajun on 1/13/17.
//
#include "real_time_monitor.hpp"
#include<iostream>
using namespace std;

RealTimeMonitor::RealTimeMonitor(string a, MultiTargetDetector &d, ClassIndependentTracker &t)
        :address(a), detector(d), tracker(t), runStatus(false), stopSignal(false){}

void RealTimeMonitor::loop(){
    cout<<"start detect and track loop"<<endl;
    while(!stopSignal) {
        Mat preImage = getCurrentImage();
        Mat curImage = getUpdatedImage();
        detectTrack(preImage, curImage);
    }
    runStatus = false;
    stopSignal = false;
    cout<<"end detect and track loop"<<endl;
}
bool RealTimeMonitor::isRunning() const {
    return runStatus;
}
void RealTimeMonitor::run(){ // two threads call run() at the same time may cause error
    if(runStatus) return;
    runStatus = true; stopSignal = false;
    if(!cap.isOpened()) cap.open(address);
    thread loopThread(&RealTimeMonitor::loop, this);
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
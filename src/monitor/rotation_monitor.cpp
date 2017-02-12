//
// Created by dujiajun on 2/11/17.
//
#include "monitor/rotation_monitor.hpp"

RotationMonitor::RotationMonitor(string a, MultiTargetDetector &d, ClassIndependentTracker &t)
    :RealTimeMonitor(a, d, t){}

void RotationMonitor::detectTrackLoop() {
    int cnt=0;
    while(!stopSignal) {
        if(cnt==0) detect();
        else track();
        this_thread::sleep_for(chrono::milliseconds(10));
        cnt++;
        if(cnt==20) cnt=0;
    }
}

void RotationMonitor::detect(){
    cout<<"detecting ..."<<endl;
    Mat curImage = getUpdatedImage();
    if(!curImage.empty()) {
        targets = detector.detectTargets(curImage);
        for (Target t: targets) {
            t.setImage(curImage);
        }
    }
}

void RotationMonitor::track(){
    cout<<"tracking ..."<<endl;
    for(Target &t: targets){
        Mat preImage = t.getImage(), curImage = getUpdatedImage();
        if(curImage.empty()) continue;
        Rect preRegion = t.getRegion();
        t.setImage(curImage);
        t.setRegion(tracker.getUpdateRegion(preImage, curImage, preRegion));
    }
}


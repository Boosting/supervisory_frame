//
// Created by dujiajun on 1/13/17.
//
#include<iostream>
#include<thread>
#include "real_time_monitor.hpp"
using namespace std;

RealTimeMonitor::RealTimeMonitor(MultiTargetDetector d, ClassIndependentTracker t):detector(d), tracker(t) {}

void loop(RealTimeMonitor *monitor){
    int cnt=0;
    while(monitor->isRunning()) {
        if(cnt==0) monitor->detect();
        else monitor->track();
        this_thread::sleep_for(chrono::milliseconds(10));
        cnt++;
        if(cnt==20) cnt=0;
    }
}
bool RealTimeMonitor::isRunning() const {
    return runStatus;
}
void RealTimeMonitor::run(){
    if(runStatus) return;
    thread loopThread(loop, this);
    loopThread.detach();
}
void RealTimeMonitor::stop(){
    runStatus=false;
}
void RealTimeMonitor::detect(){
    Mat curImage = getCurrentImage();
    targets = detector.detectTargets(curImage);
    for(Target t: targets){
        t.setImage(curImage);
    }
}
void RealTimeMonitor::track(){
    for(Target &t: targets){
        Mat preImage = t.getImage(), curImage = getCurrentImage();
        Rect preRegion = t.getRegion();
        t.setImage(curImage);
        t.setRegion(tracker.getUpdateRegion(preImage, curImage, preRegion));
    }
}

int main(){
    MultiTargetDetector detector;
    ClassIndependentTracker tracker;
    RealTimeMonitor monitor(detector, tracker);
    monitor.run();
    cout<<"start"<<endl;
    for(long long i=0;i<10000000000;i++){
        if(i%100000000==0) cout<<i<<endl;
    }
    while(true){}
}
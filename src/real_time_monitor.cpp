//
// Created by dujiajun on 1/13/17.
//
#include "real_time_monitor.hpp"
#include<iostream>

using namespace std;

RealTimeMonitor::RealTimeMonitor(string a, DetectTrackFusion &detectTrackFusion, Displayer &dis)
        :address(a), fusion(detectTrackFusion), displayer(dis), runStatus(false), stopSignal(false){}

void RealTimeMonitor::loop(){
    cout<<"start detect and track loop"<<endl;
    Mat preImage, curImage;
    vector<Target> preTargets, curTargets;
    cap>>curImage;
    displayer.run();
    while(!stopSignal) {
        preImage = curImage;
        cap>>curImage;
        preTargets = curTargets;
        if(preImage.empty()||curImage.empty()) break;
        curTargets = fusion.detectTrack(preImage, curImage, preTargets);
        for(Target &target: curTargets) {
            Rect region = target.getRegion();
            int x1 = region.x, y1 = region.y;
            int w = region.width, h = region.height;
            int x2 = x1+w-1, y2 = y1+h-1;
            double score = target.getScore();
            cout << "x1: " << x1 << " y1: " << y1 << " x2: " << x2 << " y2: " << y2 << endl;
            cout << "score: " << score << endl;
        }
        displayer.setImage(curImage);
        displayer.setTargets(curTargets);
    }
    displayer.stop();
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
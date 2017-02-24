//
// Created by dujiajun on 2/23/17.
//

#include "displayer.hpp"
Displayer::Displayer():runStatus(false), stopSignal(false){}

void Displayer::run(){
    if(runStatus) return;
    runStatus = true; stopSignal = false;
    thread loopThread(&Displayer::loop, this);
    loopThread.detach();
}

void Displayer::stop(){
    if(!runStatus) return;
    stopSignal = true;
}

void Displayer::loop(){
    int i=0;
    while(!stopSignal) {
        Mat tmpImage = getImage();
        vector<Target> tmpTargets = getTargets();
        if(tmpImage.empty()) continue;
        for(Target &target: tmpTargets){
            Rect region = target.getRegion();
            rectangle(tmpImage, region, Scalar( 255, 0, 0 ), 1, 1);
        }
        imwrite("/home/dujiajun/detect_result/" + to_string(i++) + ".jpg", tmpImage);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    runStatus = false;
    stopSignal = false;
}

void Displayer::setImage(Mat img){
    boost::unique_lock<boost::shared_mutex> lock(image_mutex);
    curImage = img;
}

Mat Displayer::getImage() const{
    boost::shared_lock<boost::shared_mutex> lock(image_mutex);
    Mat curImageClone = curImage.clone();
    return curImageClone;
}

void Displayer::setTargets(vector<Target> targetVec){
    boost::unique_lock<boost::shared_mutex> lock(targets_mutex);
    targets = targetVec;
}

vector<Target> Displayer::getTargets() const{
    boost::shared_lock<boost::shared_mutex> lock(targets_mutex);
    return targets;
}

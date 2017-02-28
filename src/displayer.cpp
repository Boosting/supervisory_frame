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
    map<Target::TARGET_CLASS, String> targetClassName;
    targetClassName[Target::PERSON] = "person";
    targetClassName[Target::BICYCLE] = "bicycle";
    targetClassName[Target::BUS] = "bus";
    targetClassName[Target::CAR] = "car";
    targetClassName[Target::MOTORBIKE] = "motorbike";
    targetClassName[Target::TRUCK] = "truck";

    while(!stopSignal) {
        Mat tmpImage = getImage();
        vector<Target> tmpTargets = getTargets();
        if(tmpImage.empty()) continue;
        for(Target &target: tmpTargets){
            Target::TARGET_CLASS targetClass = target.getClass();
            auto it = targetClassName.find(targetClass);
            if(it!=targetClassName.end()){
                String name = it->second;
                Rect region = target.getRegion();
                String id(to_string(target.getId()));
                int x = region.x, y = region.y;
                rectangle(tmpImage, region, Scalar(255, 0, 0), 3, 1);
                putText(tmpImage, name, Point(x+5, y+30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
                putText(tmpImage, id, Point(x+5, y+60), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1);
            }
        }
        imshow("displayer", tmpImage);
        waitKey(100);
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

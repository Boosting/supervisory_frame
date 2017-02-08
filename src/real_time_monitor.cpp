//
// Created by dujiajun on 1/13/17.
//
#include<iostream>
#include "real_time_monitor.hpp"
using namespace std;

RealTimeMonitor::RealTimeMonitor(string a, MultiTargetDetector d, ClassIndependentTracker t):address(a), detector(d), tracker(t), runStatus(false) {}

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
	cap.open(address);
	runStatus = true;
    thread loopThread(loop, this);
    loopThread.detach();
}
void RealTimeMonitor::stop(){
    runStatus=false;
}
/**
 * public function
 *
 * @return
 */
Mat RealTimeMonitor::getCurrentImage(){
    boost::shared_lock<boost::shared_mutex> lock(image_mutex);
    return currentImage;
}
/**
 * private function
 * get real current image and update it
 * only main loop thread can update currentImage
 * @return
 */
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
    targets = detector.detectTargets(curImage);
    for(Target t: targets){
        t.setImage(curImage);
    }
}
void RealTimeMonitor::track(){
    for(Target &t: targets){
        Mat preImage = t.getImage(), curImage = getUpdatedImage();
        Rect preRegion = t.getRegion();
        t.setImage(curImage);
        t.setRegion(tracker.getUpdateRegion(preImage, curImage, preRegion));
    }
}

int main(){
    string address="/home/dujiajun/car_person_video.mp4";
    string model_file="/home/dujiajun/py-faster-rcnn/models/kitti/VGG16/faster_rcnn_end2end/test.prototxt";
    string trained_file="/home/dujiajun/py-faster-rcnn/data/kitti/VGG16/faster_rcnn_end2end.caffemodel";
    MultiTargetDetector detector(model_file, trained_file);
    ClassIndependentTracker tracker;
    RealTimeMonitor monitor(address, detector, tracker);
    monitor.run();
    while (monitor.isRunning())
    {
        Mat frame = monitor.getCurrentImage();
        vector<Target> targets = monitor.getTargets();
        for(Target target: targets){
            Rect region = target.getRegion();
			rectangle(frame, region, Scalar( 255, 0, 0 ), 2, 1);
        }
        imshow("monitor", frame);
        if (cvWaitKey(10) == 'q')
            break;
    }
}

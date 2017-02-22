//
// Created by dujiajun on 2/11/17.
//
#include "monitor/rotation_monitor.hpp"

RotationMonitor::RotationMonitor(string a, MultiTargetDetector &d, ClassIndependentTracker &t)
    :RealTimeMonitor(a, d, t){}

void RotationMonitor::detectTrackLoop() {
    while(!stopSignal) {
        Mat preImage = getCurrentImage();
        Mat curImage = getUpdatedImage();
        map<unsigned long long, Target> detectMap = detect(curImage);
        map<unsigned long long, Target> trackMap = track(curImage, preImage);
        map<unsigned long long, Target> fusionMap;
        for(auto &pair: detectMap){
            unsigned long long id = pair.first;
            Target &detectTarget = pair.second;
            auto it = trackMap.find(id);
            if(it!=trackMap.end()) {
                Target &trackTarget = it->second;
                //fusion
            } else {
                fusionMap[id] = detectTarget;
            }
        }
        for(auto &pair: trackMap) {
            unsigned long long id = pair.first;
            Target &trackTarget = pair.second;
            if(fusionMap.find(id)==fusionMap.end()){
                fusionMap[id] = trackTarget;
            }
        }
        vector<Target> updatedTargets(fusionMap.size());
        int i=0;
        for(auto &pair: fusionMap){
            unsigned long long id = pair.first;
            Target &target = pair.second;
            target.setId(id);
            updatedTargets[i] = target;
            i++;
        }
        targets = updatedTargets;
        this_thread::sleep_for(chrono::milliseconds(10));
    }
}

map<unsigned long long, Target> RotationMonitor::detect(const Mat curImage){
    cout<<"detecting ..."<<endl;
    targets = detector.detectTargets(curImage);
    for (Target t: targets) {
        //find id
    }
}

map<unsigned long long, Target> RotationMonitor::track(const Mat curImage, const Mat preImage){
    cout<<"tracking ..."<<endl;
    map<unsigned long long, Target> trackTargets;
    for(Target &target: targets){
        if(target.getScore()<0.5) continue; //if score is low, don't perform track
        unsigned long long id = target.getId();
        Target::TARGET_CLASS target_class = target.getClass();
        Rect preRegion = target.getRegion();
        Rect curRegion = tracker.getUpdateRegion(preImage, curImage, preRegion);
        trackTargets[id] = Target(curRegion, target_class, 1.0, id);
    }
    return trackTargets;
}


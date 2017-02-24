//
// Created by dujiajun on 2/11/17.
//
#include <set>
#include "monitor/rotation_monitor.hpp"

RotationMonitor::RotationMonitor(string a, MultiTargetDetector &det, ClassIndependentTracker &tra, Displayer &dis)
    :RealTimeMonitor(a, det, tra, dis){}

vector<Target> RotationMonitor::detectTrack(Mat preImage, Mat curImage, vector<Target> preTargets) {
    clock_t t1=clock();
	map<unsigned long long, Target> detectMap = detect(curImage, preTargets);
	clock_t t2=clock();
	map<unsigned long long, Target> trackMap = track(preImage, curImage, preTargets);
    clock_t t3=clock();
	map<unsigned long long, Target> fusionMap;
    for(auto &pair: detectMap){
        unsigned long long id = pair.first;
        Target &detectTarget = pair.second;
        auto it = trackMap.find(id);
        if(it!=trackMap.end()) {
            Target &trackTarget = it->second;
            //do fusion to detect and track result
            fusionMap[id] = detectTarget; // write fusion algorithm later
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
	clock_t t4=clock();
	cout<<"detect use time: "<<double(t2-t1)/CLOCKS_PER_SEC<<endl;
	cout<<"track use time: "<<double(t3-t2)/CLOCKS_PER_SEC<<endl;
	cout<<"fusion use time: "<<double(t4-t3)/CLOCKS_PER_SEC<<endl;
    return updatedTargets;
}

double RotationMonitor::getOverlapRate(Rect r1, Rect r2){
    if(r1.area()<=0||r2.area()<=0) return 0;
    double overlapRate=0;
    int x1=max(r1.x, r2.x), y1=max(r1.y, r2.y);
    int x2=min(r1.x+r1.width, r2.x+r2.width);
    int y2=min(r1.y+r1.height, r2.y+r2.height);
    if(x2>=x1 && y2>=y1) {
        double overlapArea = (x2 - x1) * (y2 - y1);
        overlapRate = overlapArea / (r1.area() + r2.area() - overlapArea);
    }
    return overlapRate;
}

map<unsigned long long, Target> RotationMonitor::detect(Mat curImage, vector<Target> preTargets){
    cout<<"detecting ..."<<endl;
    map<unsigned long long, Target> detectTargets;
    vector<Target> detected = detector.detectTargets(curImage);
    set<unsigned long long> idUsed;
    for(Target &target: preTargets) {
        idUsed.insert(target.getId());
    }
    vector<bool> idGotten(detected.size(), false);
    vector<double> overlapVec(detected.size(), 0.0);

    // get id from previous targets
    double overlapThresh = 0.3; //only overlap rate > overlapThresh can be seen as the same object
    for (Target &t1: preTargets) {
        unsigned long long id = t1.getId();
        Rect r1 = t1.getRegion();
        Target::TARGET_CLASS cls1 = t1.getClass();
        int indice=-1;
        double maxOverlapRate=0;
        for(int i=0;i<detected.size();i++) {
            Target &t2 = detected[i];
            Rect r2 = t2.getRegion();
            Target::TARGET_CLASS cls2 = t2.getClass();
            if(cls2!=cls1) continue;
            double overlapRate = getOverlapRate(r1, r2);
            if(overlapRate>max(overlapThresh, max(maxOverlapRate, overlapVec[i]))) {
                indice = i;
                maxOverlapRate = overlapRate;
                overlapVec[i] = overlapRate;
            }
        }
        if(indice!=-1) {
            idGotten[indice] = true;
            detected[indice].setId(id);
        }
    }

    //get a random id for target without id
    default_random_engine generator;
    uniform_int_distribution<int> distribution(1, 1000000000);
    auto dice = bind(distribution, generator);
    for(int i=0;i<detected.size();i++){
        if(idGotten[i]) continue;
        unsigned long long id;
        do {
            id = dice();
        } while(idUsed.find(id)!=idUsed.end());
        detected[i].setId(id);
        idUsed.insert(id);
    }

    for(Target &target: detected) {
        detectTargets[target.getId()] = target;
    }
    return detectTargets;
}

map<unsigned long long, Target> RotationMonitor::track(Mat preImage, Mat curImage, vector<Target> preTargets){
    cout<<"tracking ..."<<endl;
    map<unsigned long long, Target> trackTargets;
    for(Target &target: preTargets){
        if(target.getScore()<0.5) continue; //if score is low, don't perform track
        unsigned long long id = target.getId();
        Target::TARGET_CLASS target_class = target.getClass();
        Rect preRegion = target.getRegion();
        Rect curRegion = tracker.getUpdateRegion(preImage, curImage, preRegion);
        trackTargets[id] = Target(curRegion, target_class, 1.0, id);
    }
    return trackTargets;
}


//
// Created by dujiajun on 3/3/17.
//

#include "fusion/detector_only_fusion.hpp"
#include "utils/opencv_util.hpp"
#include <set>
#include <functional>

using namespace std;

DetectorOnlyFusion::DetectorOnlyFusion(MultiTargetDetector &multiTargetDetector)
        :detector(multiTargetDetector){}

vector<Target> DetectorOnlyFusion::detectTrack(Mat preImage, Mat curImage, vector<Target> preTargets){
    clock_t t1=clock();
    vector<Target> curTargets = detector.detectTargets(curImage);
    clock_t t2=clock();
    cout<<"detect use time: "<<double(t2-t1)/CLOCKS_PER_SEC<<endl;

    set<unsigned long long> idUsed;
    for(Target &target: preTargets) {
        idUsed.insert(target.getId());
    }
    vector<bool> idGotten(curTargets.size(), false);
    vector<double> overlapVec(curTargets.size(), 0.0);

    // get id from previous targets
    double overlapThresh = 0.3; //only overlap rate > overlapThresh can be seen as the same object
    for (Target &t1: preTargets) {
        unsigned long long id = t1.getId();
        Rect r1 = t1.getRegion();
        Target::TARGET_CLASS cls1 = t1.getClass();
        int indice=-1;
        double maxOverlapRate=0;
        for(int i=0;i<curTargets.size();i++) {
            Target &t2 = curTargets[i];
            Rect r2 = t2.getRegion();
            Target::TARGET_CLASS cls2 = t2.getClass();
            if(cls2!=cls1) continue;
            double overlapRate = OpencvUtil::getOverlapRate(r1, r2);
            if(overlapRate>max(overlapThresh, max(maxOverlapRate, overlapVec[i]))) {
                indice = i;
                maxOverlapRate = overlapRate;
                overlapVec[i] = overlapRate;
            }
        }
        if(indice!=-1) {
            idGotten[indice] = true;
            curTargets[indice].setId(id);
        }
    }

    //get a random id for target without id
    default_random_engine generator;
    uniform_int_distribution<int> distribution(1, 1000000000);
    auto dice = bind(distribution, generator);
    for(int i=0;i<curTargets.size();i++){
        if(idGotten[i]) continue;
        unsigned long long id;
        do {
            id = dice();
        } while(idUsed.find(id)!=idUsed.end());
        curTargets[i].setId(id);
        idUsed.insert(id);
    }
    return curTargets;
}

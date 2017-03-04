//
// Created by dujiajun on 3/3/17.
//

#include "fusion/fast_rcnn_fusion.hpp"
#include "detector/fast_rcnn_detector.hpp"
FastRcnnFusion::FastRcnnFusion()
        :detector(FastRcnnDetector()) {}

vector<Target> FastRcnnFusion::detectTrack(Mat preImage, Mat curImage, vector<Target> preTargets) {
    clock_t t1=clock();
    vector<Target> targets = detector.detectTargets(curImage);
    clock_t t2=clock();
    cout<<"detect use time: "<<double(t2-t1)/CLOCKS_PER_SEC<<endl;
    return targets;
}
//
// Created by dujiajun on 3/15/17.
//

#include <opencv/cv.hpp>
#include "motion_detector/background_substraction_motion_detector.hpp"
#include "utils/selective_search.hpp"
#include "utils/opencv_util.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
int getFrameId(string &image_name){
    string idStr = image_name.substr(15, 5);
    int frameId = atoi(idStr.c_str());
    return frameId;
}

int main() {
    string address = "/home/dujiajun/CUHKSquare.mpg";
    string bbox_file_address = "/home/dujiajun/CUHK/train_bbox.txt";
    string output_address = "/home/dujiajun/CUHK/train_output.txt";
    ifstream in(bbox_file_address);
    ofstream out(output_address);
    VideoCapture cap(address);
    Mat image;
    BackgroundSubstractionMotionDetector motion_detector;

    int frame_id = 0, prev_frame_id = -1;
    string image_name;
    int bbox_num;

    int frameNum = 0;
    int proposalNum = 0;
    int groundTrueNum = 0;
    vector<int> iou_vec1(101, 0); // 0, 0.05, 0.10 ... 0.95, 1.00
    vector<int> iou_vec2(101, 0); // 0, 0.05, 0.10 ... 0.95, 1.00
    while (in >> image_name) {
        frameNum++;
        vector<Rect> groundTrue;
        vector<Rect> proposals1;
        vector<Rect> proposals2;
        vector<Rect> regions;
        in >> bbox_num;
        groundTrueNum += bbox_num;
        for(int i=0;i<bbox_num;i++){
            int x, y, w, h;
            in>>x>>y>>w>>h;
            groundTrue.push_back({x, y, w, h});
        }
        frame_id = getFrameId(image_name);
        cout<<frame_id<<endl;
        for (int i = prev_frame_id + 1; i <= frame_id; i++) {
            cap >> image;
            regions = motion_detector.detect(image);
        }

        for (Rect &region: regions) {
            proposals1.push_back(region);
            proposals2.push_back(region);
            if(region.area()<1000){
                continue;
            }
            Mat partImage;
            int x1 = region.x, y1 = region.y;
            int w = region.width, h = region.height;
            int center_x = x1 + w / 2, center_y = y1 + h / 2;
            getRectSubPix(image, {w, h}, {center_x, center_y}, partImage);
            vector<Rect> partProposals = region_proposal::selectiveSearch(partImage, 500, 0.8, 50, 100, 100000, 2.5);
            for (Rect &partProposal: partProposals) {
                int x = x1 + partProposal.x, y = y1 + partProposal.y;
                Rect globalProposal(x, y, partProposal.width, partProposal.height);
                proposals1.push_back(globalProposal);
            }
        }
        vector<float> overlapVec1(groundTrue.size(), 0);
        vector<float> overlapVec2(groundTrue.size(), 0);
        for(Rect proposal: proposals1){
            proposal = {proposal.x*2, proposal.y*2, proposal.width*2, proposal.height*2};
            float max_overlap_rate = 0;
            int index = -1;
            for(int i=0;i<groundTrue.size();i++){
                Rect &trueRegion = groundTrue[i];
                double overlapRate = OpencvUtil::getOverlapRate(trueRegion, proposal);
                if(overlapRate>max_overlap_rate){
                    index = i;
                    max_overlap_rate = overlapRate;
                }
            }
            if(index!=-1){
                overlapVec1[index] = max(max_overlap_rate, overlapVec1[index]);
            }
        }
        for(float prob: overlapVec1){
            for(int i=0;i<min(int(prob*100-0.000001+1), 101);i++){
                iou_vec1[i]++;
            }
        }
        for(Rect proposal: proposals2){
            proposal = {proposal.x*2, proposal.y*2, proposal.width*2, proposal.height*2};
            float max_overlap_rate = 0;
            int index = -1;
            for(int i=0;i<groundTrue.size();i++){
                Rect &trueRegion = groundTrue[i];
                double overlapRate = OpencvUtil::getOverlapRate(trueRegion, proposal);
                if(overlapRate>max_overlap_rate){
                    index = i;
                    max_overlap_rate = overlapRate;
                }
            }
            if(index!=-1){
                overlapVec2[index] = max(max_overlap_rate, overlapVec2[index]);
            }
        }
        for(float prob: overlapVec2){
            for(int i=0;i<min(int(prob*100-0.000001+1), 101);i++){
                iou_vec2[i]++;
            }
        }


        prev_frame_id = frame_id;
    }
    out<<"frame num: "<<endl;
    out<<frameNum<<endl;
    out<<"ground truth num: "<<endl;
    out<<groundTrueNum<<endl;
    out<<"proposal num: "<<endl;
    out<<proposalNum<<endl;
    out<<"proposal overlap1: "<<endl;
    for(int i=0;i<iou_vec1.size();i++){
        out<<iou_vec1[i]<<" ";
    }
    out<<endl;
    out<<"proposal overlap2: "<<endl;
    for(int i=0;i<iou_vec2.size();i++){
        out<<iou_vec2[i]<<" ";
    }
    out<<endl;
    in.close();
    out.close();
    return 0;
}
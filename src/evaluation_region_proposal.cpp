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
double getGroundTruthOverlapRate(Rect groundTruth, Rect proposal){
    if(groundTruth.area()<=0||proposal.area()<=0) return 0;
    double overlapRate=0;
    int x1=max(groundTruth.x, proposal.x), y1=max(groundTruth.y, proposal.y);
    int x2=min(groundTruth.x+groundTruth.width, proposal.x+proposal.width);
    int y2=min(groundTruth.y+groundTruth.height, proposal.y+proposal.height);
    if(x2>=x1 && y2>=y1) {
        double overlapArea = (x2 - x1) * (y2 - y1);
        overlapRate = overlapArea / groundTruth.area();
    }
    return overlapRate;
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
    vector<int> proposalNumArr(100, 0); // 0, 0.05, 0.10 ... 0.95
    vector<int> proposalVec2(100, 0); // 0, 0.05, 0.10 ... 0.95
    while (in >> image_name) {
        frameNum++;
        vector<Rect> groundTrue;
        vector<Rect> proposals;
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
            if(region.area()<1000){
                proposals.push_back(region);
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
                proposals.push_back(globalProposal);
            }
        }
        for(Rect &trueRegion: groundTrue){
            vector<bool> hasOverlap(100, false);
            vector<bool> hasOverlap2(100, false);
            for(Rect &proposal: proposals){
                Rect tmpProposal = {proposal.x*2, proposal.y*2, proposal.width*2, proposal.height*2};
                double overlapRate = OpencvUtil::getOverlapRate(trueRegion, tmpProposal);
                double rate = 0.0;
                for(int i=0;i<100;i++){
                    if(rate>overlapRate) break;
                    hasOverlap[i] = true;
                    rate+=0.01;
                }

                double overlapRate2 = getGroundTruthOverlapRate(trueRegion, tmpProposal);
                rate = 0.0;
                for(int i=0;i<100;i++){
                    if(rate>overlapRate2) break;
                    hasOverlap2[i] = true;
                    rate+=0.01;
                }
            }
            for(int i=0;i<hasOverlap.size();i++) {
                if(hasOverlap[i]) proposalNumArr[i]++;
            }
            for(int i=0;i<hasOverlap2.size();i++){
                if(hasOverlap2[i]) proposalVec2[i]++;
            }
        }

        proposalNum += proposals.size();
//        for (Rect &proposal: proposals) {
//            rectangle(image, proposal, Scalar(255, 0, 0), 1, 1);
//        }
//        imshow("proposal", image);
//        waitKey(10);
        prev_frame_id = frame_id;
    }
    out<<"frame num: "<<endl;
    out<<frameNum<<endl;
    out<<"ground truth num: "<<endl;
    out<<groundTrueNum<<endl;
    out<<"proposal num: "<<endl;
    out<<proposalNum<<endl;
    out<<"proposal overlap: "<<endl;
    for(int i=0;i<proposalNumArr.size();i++){
        out<<proposalNumArr[i]<<" ";
    }
    out<<endl;
    out<<"ground truth overlap: "<<endl;
    for(int i=0;i<proposalVec2.size();i++){
        out<<proposalVec2[i]<<" ";
    }
    out<<endl;
    in.close();
    out.close();
    return 0;
}
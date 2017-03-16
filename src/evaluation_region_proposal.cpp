//
// Created by dujiajun on 3/15/17.
//

#include <opencv/cv.hpp>
#include "motion_detector/background_substraction_motion_detector.hpp"
#include "utils/selective_search.hpp"
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
    while (in >> image_name) {
        vector<Rect> groundTrue;
        vector<Rect> proposals;
        vector<Rect> regions;
        in >> bbox_num;
        for(int i=0;i<bbox_num;i++){
            int x, y, w, h;
            in>>x>>y>>w>>h;
            groundTrue.push_back({x, y, w, h});
        }
        frame_id = getFrameId(image_name);
        for (int i = prev_frame_id + 1; i <= frame_id; i++) {
            cap >> image;
            regions = motion_detector.detect(image);
        }
        int proposal_num = 0;
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
//        for (Rect &proposal: proposals) {
//            rectangle(image, proposal, Scalar(255, 0, 0), 1, 1);
//        }
//        imshow("proposal", image);
//        waitKey(10);
        prev_frame_id = frame_id;
    }
    in.close();
    out.close();
}
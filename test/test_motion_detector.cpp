//
// Created by dujiajun on 3/5/17.
//

#include <opencv/cv.hpp>
#include "motion_detector/background_substraction_motion_detector.hpp"
#include "utils/selective_search.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
//    string address="rtsp://192.168.1.201:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream";
    string address = "/home/dujiajun/CUHKSquare.mpg";
    VideoCapture cap(address);
    Mat image;
    BackgroundSubstractionMotionDetector motion_detector;
    while(cap.read(image)){
        for(int i=0;i<20;i++) cap>>image;
        vector<Rect> regions = motion_detector.detect(image);
        int proposal_num = 0;
        for(Rect &region: regions){
            Mat partImage;
            int x1 = region.x, y1 = region.y;
            int w = region.width, h = region.height;
            int center_x = x1 + w / 2, center_y = y1 + h / 2;
            getRectSubPix(image, {w, h}, {center_x, center_y}, partImage);
            vector<Rect> proposals = region_proposal::selectiveSearch(partImage, 500, 0.8, 50, 100, 100000, 2.5 );
            proposal_num += proposals.size();
            for(Rect &proposal: proposals){
                int x = x1 + proposal.x, y = y1 + proposal.y;
                rectangle(image, {x, y, proposal.width, proposal.height}, Scalar(255, 0, 0), 1, 1);
            }
        }
        cout<<"proposal num: "<<proposal_num<<endl;
        imshow("motion detector", image);
        waitKey(10);
    }
}

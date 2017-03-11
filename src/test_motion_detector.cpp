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
    VideoCapture cap("/home/dujiajun/CUHKSquare.mpg");
    Mat image;
    BackgroundSubstractionMotionDetector motion_detector;
    while(cap.read(image)){
        vector<Rect> regions = motion_detector.detect(image);
        int proposal_num = 0;
        for(Rect &region: regions){
            Mat partImage;
            int x1 = region.x, y1 = region.y;
            int w = region.width, h = region.height;
            int center_x = x1 + w / 2, center_y = y1 + h / 2;
            getRectSubPix(image, {w, h}, {center_x, center_y}, partImage);
            vector<Rect> proposals = region_proposal::selectiveSearch(partImage, 500, 0.8, 50, 1000, 100000, 2.5 );
            proposal_num += proposals.size();
            for(Rect &proposal: proposals){
                int x = x1 + proposal.x, y = y1 + proposal.y;
                rectangle(image, {x, y, proposal.width, proposal.height}, Scalar(255, 0, 0), 3, 1);
            }
        }
        cout<<"proposal num: "<<proposal_num<<endl;
        imshow("motion detector", image);
        waitKey(5);
    }
}

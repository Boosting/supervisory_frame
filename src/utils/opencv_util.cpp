//
// Created by dujiajun on 3/4/17.
//

#include "utils/opencv_util.hpp"

double OpencvUtil::getOverlapRate(Rect r1, Rect r2){
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

void OpencvUtil::makeRegionsVaild(vector<Rect> &regions, const Mat &image){
    int image_width = image.cols, image_height = image.rows;
    for(Rect &region: regions){
        region.x = max(region.x, 0);
        region.y = max(region.y, 0);
        region.width = min(region.width, image_width - region.x);
        region.height = min(region.height, image_height - region.y);
    }
}
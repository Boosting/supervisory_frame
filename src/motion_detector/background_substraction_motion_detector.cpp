//
// Created by dujiajun on 3/5/17.
//

#include <queue>
#include "motion_detector/background_substraction_motion_detector.hpp"

BackgroundSubstractionMotionDetector::BackgroundSubstractionMotionDetector(){
    mog = createBackgroundSubtractorMOG2();
}

vector<Rect> BackgroundSubstractionMotionDetector::detect(const Mat &image){
    Mat GaussianImage, foreground;
    GaussianBlur(image, GaussianImage, {11, 11}, 2);
    mog->apply(GaussianImage, foreground, 0.001);
    erode(foreground, foreground, cv::Mat());
    dilate(foreground, foreground, cv::Mat());
    vector<vector<bool> > isForeground = getBinaryImage(foreground, 100);
    vector<Rect> regions = getRegions(isForeground);
    return regions;
}
vector<vector<bool> > BackgroundSubstractionMotionDetector::getBinaryImage(const Mat &image, int thresh){
    int height = image.rows, width = image.cols;
    vector<vector<bool> > binaryImage(width, vector<bool>(height, false));
    for(int j=0;j<height;j++) //may need speed up
    {
        const uchar *data = image.ptr<uchar>(j);
        for(int i=0;i<width;i++){
            binaryImage[i][j] = (data[i] >= thresh);
        }
    }
    return binaryImage;
}
vector<Rect> BackgroundSubstractionMotionDetector::getRegions(const vector<vector<bool> > &isForeground){
    vector<Rect> regions;
    if(isForeground.empty()||isForeground[0].empty()) return regions;
    int width = isForeground.size(), height = isForeground[0].size();
    vector<vector<bool> > visited(width, vector<bool>(height, false));
    for(int i=0;i<width;i++){
        for(int j=0;j<height;j++){
            if(!isForeground[i][j] || visited[i][j]) continue;
            queue<vector<int> > que;
            que.push({i, j});
            int x1=i, x2=i, y1=j, y2=j;
            int region_size = 0;
            while(!que.empty()){
                vector<int> &xy=que.front();
                int x=xy[0], y=xy[1];
                que.pop();
                if(x<0 || x>=width || y<0 || y>=height) continue;
                if(!isForeground[x][y] || visited[x][y]) continue;
                visited[x][y] = true;
                region_size++;
                x1=min(x1, x), x2=max(x2, x);
                y1=min(y1, y), y2=max(y2, y);
                que.push({x, y+1});
                que.push({x, y-1});
                que.push({x+1, y});
                que.push({x-1, y});
                que.push({x-1, y-1});
                que.push({x-1, y+1});
                que.push({x+1, y-1});
                que.push({x+1, y+1});
            }
            Rect region(x1, y1, x2-x1+1, y2-y1+1);
            if(region_size>=1000){
                regions.push_back(region);
            }
        }
    }
    return regions;
}
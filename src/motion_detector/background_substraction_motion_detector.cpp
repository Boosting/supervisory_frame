//
// Created by dujiajun on 3/5/17.
//

#include <queue>
#include "motion_detector/background_substraction_motion_detector.hpp"

BackgroundSubstractionMotionDetector::BackgroundSubstractionMotionDetector(){
    mog = createBackgroundSubtractorMOG2(20, 16, true);
}

vector<Rect> BackgroundSubstractionMotionDetector::detect(const Mat &image){
    Mat gaussianImage, foreground;
    GaussianBlur(image, gaussianImage, {5, 5}, 2);
    mog->apply(gaussianImage, foreground, 0.001);
    erode(foreground, foreground, cv::Mat());
    dilate(foreground, foreground, cv::Mat());
//    imshow("motion detector", foreground);
//    waitKey(5);
    vector<vector<int> > foregroundMask = getForegroundMask(foreground);
    vector<Rect> regions = getRegions(foregroundMask);
    return regions;
}
vector<vector<int> > BackgroundSubstractionMotionDetector::getForegroundMask(const Mat &image){
    // 0: background, 1: shadow, 2: foreground
    int height = image.rows, width = image.cols;
    vector<vector<int> > binaryImage(width, vector<int>(height, false));
    for(int j=0;j<height;j++) //may need speed up
    {
        const uchar *data = image.ptr<uchar>(j);
        for(int i=0;i<width;i++){
            int mask = 0;
            switch(data[i]){
                case 0: mask=0; break;
                case 255: mask=2; break;
                default: mask=1; break;
            }
            binaryImage[i][j] = mask;
        }
    }
    return binaryImage;
}
vector<Rect> BackgroundSubstractionMotionDetector::getRegions(vector<vector<int> > &foregroundMask){
    vector<Rect> regions;
    if(foregroundMask.empty()||foregroundMask[0].empty()) return regions;
    int width = foregroundMask.size(), height = foregroundMask[0].size();
    vector<Point> points;
    vector<vector<bool> > visited(width, vector<bool>(height, false));
    for(int i=0;i<width;i++){
        for(int j=0;j<height;j++){
            if(foregroundMask[i][j]!=2 || visited[i][j]) continue;
            queue<vector<int> > que;
            que.push({i, j});
            Point init_point(i, j);
            int x1=i, x2=i, y1=j, y2=j;
            int region_size = 0;
            while(!que.empty()){
                vector<int> &xy=que.front();
                int x=xy[0], y=xy[1];
                que.pop();
                if(x<0 || x>=width || y<0 || y>=height) continue;
                if(foregroundMask[x][y]!=2 || visited[x][y]) continue;
                visited[x][y] = true;
                region_size++;
                x1 = min(x1, x), x2 = max(x2, x);
                y1 = min(y1, y), y2 = max(y2, y);
                que.push({x, y+1});
                que.push({x, y-1});
                que.push({x+1, y});
//                que.push({x-1, y});  no need to search above
            }
            Rect region(x1, y1, x2-x1+1, y2-y1+1);
            if(region_size>=100){
                regions.push_back(region);
                points.push_back(init_point);
            }
        }
    }
    regions = expandRegions(foregroundMask, regions, points);
    return regions;
}

vector<Rect> BackgroundSubstractionMotionDetector::expandRegions(vector<vector<int> > &foregroundMask, vector<Rect> &regions, vector<Point> &points){
    int expand_thresh = 20;
    vector<Rect> expandedRegions;
    int width = foregroundMask.size(), height = foregroundMask[0].size();

    for(int i=0;i<width;i++){
        for(int j=0;j<height;j++){
            if(foregroundMask[i][j]==2){
                bool isNoise = true;
                for(Rect &region: regions){
                    int x1=region.x, y1=region.y, w=region.width, h=region.height;
                    int x2=x1+w-1, y2=y1+h-1;
                    if(i>=x1 && i<=x2 && j>=y1 && j<=y2){
                        isNoise = false; break;
                    }
                }
                if(isNoise) foregroundMask[i][j]=1;
            }
        }
    }
    vector<vector<bool> > visited(width, vector<bool>(height, false));
    for(int t=0;t<regions.size();t++){
        Rect &region = regions[t];
        int x1=region.x, y1=region.y, w=region.width, h=region.height;
        int x2=x1+w-1, y2=y1+h-1;
        bool isContained = false;
        for(Rect &expandedRegion: expandedRegions){
            int ex1=expandedRegion.x, ey1=expandedRegion.y, ew=expandedRegion.width, eh=expandedRegion.height;
            int ex2=ex1+ew-1, ey2=ey1+eh-1;
            if(x1>=ex1 && x2<=ex2 && y1>=ey1 && y2<=ey2) {
                isContained=true; break;
            }
        }
        if(isContained) continue;

        Point &init_point=points[t];
        queue<Point> que;
        que.push(init_point);
        int inner_x1=x1, inner_y1=y1, inner_x2=x2, inner_y2=y2;
        int outer_x1=x1, outer_y1=y1, outer_x2=x2, outer_y2=y2;
        while(!que.empty()){
            Point point = que.front();
            que.pop();
            int x = point.x, y = point.y;
            if(x<0 || x>=width || y<0 || y>=height) continue;
            if(visited[x][y]) continue;
            if(foregroundMask[x][y]==0) continue;
            else if(foregroundMask[x][y]==1 &&
                    (x<inner_x1-expand_thresh || x>inner_x2+expand_thresh
                    || y<inner_y1-expand_thresh || y>inner_y2+expand_thresh)) continue;
            visited[x][y] = true;
            que.push({x+1, y});
            que.push({x-1, y});
            que.push({x, y+1});
            que.push({x, y-1});
            outer_x1 = min(outer_x1, x), outer_x2 = max(outer_x2, x);
            outer_y1 = min(outer_y1, y), outer_y2 = max(outer_y2, y);
            if(foregroundMask[x][y]==2){
                inner_x1 = min(inner_x1, x), inner_x2 = max(inner_x2, x);
                inner_y1 = min(inner_y1, y), inner_y2 = max(inner_y2, y);
            }
        }
        expandedRegions.push_back({outer_x1, outer_y1, outer_x2-outer_x1+1, outer_y2-outer_y1+1});
    }

    return expandedRegions;
}
//
// Created by dujiajun on 4/3/17.
//
#include "detector/fast_rcnn_detector.hpp"

int main(){
    string address="/home/dujiajun/CUHKSquare.mpg";
    string fast_prototxt = "/home/dujiajun/fast-rcnn/models/VGG16/test.prototxt";
    string fast_trained_model = "/home/dujiajun/fast-rcnn/data/fast_rcnn_models/vgg16_fast_rcnn_iter_40000.caffemodel";
    FastRcnnDetector detector(fast_prototxt, fast_trained_model);
    VideoCapture cap(address);
    Mat image;

    string output_address = "/home/dujiajun/CUHK/evaluate_detector_output.txt";
    ofstream out(output_address);
    int frame_id = 0;
    while(cap.read(image)){
        vector<Target> targets = detector.detectTargets(image);
        for(Target &target: targets){
            if(target.getClass() == Target::PERSON){
                const Rect &region = target.getRegion();
                int x1 = region.x, y1 = region.y, w = region.width, h = region.height;
                int x2 = x1+w-1, y2 = y1+h-1;
                out<<frame_id<<" "<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" "<<target.getScore();
            }
        }
        frame_id++;
    }
    out.close();
    return 0;
}

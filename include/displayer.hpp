//
// Created by dujiajun on 2/23/17.
//

#ifndef SUPERVISORY_FRAME_DISPLAYER_HPP
#define SUPERVISORY_FRAME_DISPLAYER_HPP

#include <opencv/cv.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include<mutex>
#include<thread>
#include <atomic>
#include "target.hpp"

using namespace std;
using namespace cv;

class Displayer {
private:
    atomic_bool stopSignal;
    atomic_bool runStatus;
    Mat curImage;
    vector<Target> targets;
    mutable boost::shared_mutex image_mutex;
    mutable boost::shared_mutex targets_mutex;
    void loop();
public:
    Displayer();
    void run();
    void stop();
    void setImage(Mat img);
    Mat getImage() const;
    void setTargets(vector<Target> targetVec);
    vector<Target> getTargets() const;
};

#endif //SUPERVISORY_FRAME_DISPLAYER_HPP

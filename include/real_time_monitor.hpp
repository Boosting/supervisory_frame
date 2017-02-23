//
// Created by root on 1/23/17.
//

#ifndef VIDEOSTREAM_REAL_TIME_MONITOR_HPP
#define VIDEOSTREAM_REAL_TIME_MONITOR_HPP

#include <opencv/cv.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include<mutex>
#include<thread>
#include <atomic>
#include "target.hpp"
#include "class_independent_tracker.hpp"
#include "multi_target_detector.hpp"

using namespace std;
using namespace cv;
class RealTimeMonitor {
public:
    /**
     * @brief Constructor function.
     * @param a The IP address of the video stream.
     * @param d The detector.
     * @param t The tracker.
     */
    RealTimeMonitor(string a, MultiTargetDetector &d, ClassIndependentTracker &t);

    /**
     * @brief Judge whether the monitor is running.
     * @return Bool, whether the monitor is running.
     */
    bool isRunning() const;

    /**
     * @brief Run the monitor.
     * If the video capture is not open, open the video capture.
     * Start the detecting and tracking loop.
     */
    void run();

    /**
     * @brief Stop the monitor.
     * Attention: after call this function, the monitor doesn't stop immediately.
     */
    void stop();

    /**
     * @brief Get the current image.
     * @return The current image.
     */
    Mat getCurrentImage();

    /**
     * @brief Get the detected targets.
     * @return A vector of the detected targets.
     */
    vector<Target> getTargets();

protected:
    MultiTargetDetector &detector;
    ClassIndependentTracker &tracker;

private:
    VideoCapture cap;
    string address;
    Mat currentImage;
    mutable boost::shared_mutex image_mutex;
    mutable boost::shared_mutex targets_mutex;
    atomic_bool stopSignal;
    atomic_bool runStatus;
    vector<Target> targets;

    /**
     * @brief Get the newest image from the video stream,
     * and update currentImage.
     * @return The updated image.
     */
    Mat getUpdatedImage();

    /**
     * @brief Perform the detecting and tracking loop.
     */
    void loop();

    /**
     * @brief Implement the detecting and tracking update targets' regions method in the subclass.
     * @param preImage Previous image.
     * @param curImage Current image.
     * @param preTargets Previous targets.
     * @return Vector of targets.
     */
    virtual vector<Target> detectTrack(Mat preImage, Mat curImage, vector<Target> preTargets) = 0;

    void setTargets(vector<Target> targetVec);
};


#endif //VIDEOSTREAM_REAL_TIME_MONITOR_HPP

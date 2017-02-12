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

    /**
     * @brief Perform a round of detecting from the current image.
     */
    void detect();

    /**
     * @brief Perform a round of tracking for the detected targets.
     */
    void track();

private:
    VideoCapture cap;
    string address;
    Mat currentImage;
    mutable boost::shared_mutex image_mutex;
    atomic_bool stopSignal;
    atomic_bool runStatus;
    vector<Target> targets; // targets needs lock to prevent read operation when written
    MultiTargetDetector &detector;
    ClassIndependentTracker &tracker;

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
};


#endif //VIDEOSTREAM_REAL_TIME_MONITOR_HPP

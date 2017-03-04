//
// Created by root on 1/23/17.
//

#ifndef SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP
#define SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP
#include "target.hpp"
#include <vector>
using namespace std;

/**
 * @brief A detector that can detect targets of multiple classes.
 */
class MultiTargetDetector{
public:
    /**
     * @brief Detect the targets from the image.
     * @param image The image.
     * @return A vector of targets.
     */
    virtual vector<Target> detectTargets(const Mat& image) = 0;
protected:
    /**
     * @brief A vector of target classes, the first element is background
     */
    vector<Target::TARGET_CLASS> idToClass;

    vector<vector<vector<int> > > bbox_transform(const vector<vector<float> > &rois, const vector<vector<float> > &bbox_pred);
    vector<vector<int> > nms(const vector<vector<vector<int> > > &bbox, const vector<vector<float> > &cls_prob, float thresh = 0.7, float min_trust_score = 0.1);

    /**
     * @brief Transform bbox class vector to Target vector
     * @param bbox_cls Vector of bbox and class,
     * each item is vector of size 6: x1, y1, x2, y2, class id, score.
     * @return Vector of Target.
     */
    vector<Target> bboxToTarget(vector<vector<float> > bbox_cls_score);
};

#endif //SUPERVISORY_FRAME_MULTI_TARGET_DETECTOR_HPP

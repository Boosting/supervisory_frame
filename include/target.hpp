//
// Created by dujiajun on 1/23/17.
//

#ifndef SUPERVISORY_FRAME_TARGET_HPP
#define SUPERVISORY_FRAME_TARGET_HPP
#include <opencv/cv.hpp>
using namespace cv;

/**
 * @brief A class recording the class, region and other information of the target detected.
 */
class Target {
public:
	enum TARGET_CLASS {
		UNKNOWN,
		PERSON,
		BICYCLE,
		CAR,
		MOTORBIKE,
		BUS,
		TRAIN,
		TRUCK,
		TRAFFIC_LIGHT,
		FIRE_HYDRANT,
		STOP_SIGN
	};
private:
    /**
     * @brief The unique id associating with the target.
     * If id equals to 0, the target has not been associated with an id.
     */
	unsigned long long id;

    Rect region;

	enum TARGET_CLASS target_class;

    /**
     * @brief The score (or probability) of the target, between 0.0 and 1.0.
     */
    double score;
public:
    Target(Rect r=Rect(), TARGET_CLASS t=UNKNOWN, double s=1.0, unsigned long long i=0);
    void setId(unsigned long long i);
    unsigned long long getId() const;
    TARGET_CLASS getClass() const;
    void setClass(TARGET_CLASS t);
    Rect getRegion() const;
    void setRegion(Rect r);
    double getScore() const;
    void setScore(double s);
};

#endif //SUPERVISORY_FRAME_TARGET_HPP

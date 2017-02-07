//
// Created by root on 1/23/17.
//

#include "target.hpp"

Target::Target(TARGET_CLASS t):target_class(t) {}

TARGET_CLASS Target::getClass() const{
    return target_class;
}

void Target::setClass(TARGET_CLASS t) {
    target_class = t;
}

Rect Target::getRegion() const{
    return region;
}

void Target::setRegion(Rect r) {
    region = r;
}

Mat Target::getImage() const{
    return image;
}

void Target::setImage(Mat i) {
    image = i;
}
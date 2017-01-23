//
// Created by root on 1/23/17.
//

#include "target.hpp"

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
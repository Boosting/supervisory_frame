//
// Created by root on 1/23/17.
//

#include "target.hpp"

Target::Target(TARGET_CLASS t):target_class(t), id(0), score(1.0) {}

void Target::setId(unsigned long long i){
    id = i;
}

unsigned long long Target::getId() const{
    return id;
}

Target::TARGET_CLASS Target::getClass() const{
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

double Target::getScore() const{
    return score;
}
void Target::setScore(double s){
    score = s;
}

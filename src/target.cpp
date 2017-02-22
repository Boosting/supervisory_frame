//
// Created by root on 1/23/17.
//

#include "target.hpp"

Target::Target(Rect r, TARGET_CLASS t, double s, unsigned long long i):region(r), target_class(t), score(s), id(i) {}

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

double Target::getScore() const{
    return score;
}
void Target::setScore(double s){
    score = s;
}

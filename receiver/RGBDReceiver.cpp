//
// Created by zhanghaodan on 2022/7/11.
//

#include "RGBDReceiver.h"

char *RGBDData::getImage(int i) {
    return nullptr;
}

char *RGBDData::getDepth(int i) {
    return nullptr;
}

char *RGBDData::getMask(int i) {
    return nullptr;
}

RGBDData *RGBDReceiver::getData() {
    return nullptr;
}

RGBDReceiver::~RGBDReceiver() {

}

RGBDReceiver::RGBDReceiver() {

}

int RGBDReceiver::open(std::string filename) {
    return 0;
}

int RGBDReceiver::close() {
    return 0;
}

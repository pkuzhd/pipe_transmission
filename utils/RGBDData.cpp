//
// Created by zhanghaodan on 2022/7/19.
//

#include "RGBDData.h"

char *RGBDData::getImage(int i) {
    int bias = 0;
    for(int j = 0 ; j < i ; j++){
        bias += w[j] * h[j] * 3;
    }
    return imgs + bias;
}

char *RGBDData::getDepth(int i) {
    int bias = 0;
    for(int j = 0 ; j < i ; j++){
        bias += w_crop[j] * h_crop[j] * 4;
    }
    return depths + bias;
}

char *RGBDData::getMask(int i) {
    int bias = 0;
    for(int j = 0 ; j < i ; j++){
        bias += w_crop[j] * h_crop[j];
    }
    return masks + bias;
}
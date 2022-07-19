//
// Created by zhanghaodan on 2022/7/19.
//

#ifndef PIPE_TRANSMISSION_IRGBDRECEIVER_H
#define PIPE_TRANSMISSION_IRGBDRECEIVER_H

#include <string>
#include "RGBDData.h"

class IRGBDReceiver {
public:

    virtual int open(std::string filename) = 0;

    virtual int close() = 0;

    virtual RGBDData *getData() = 0;
};


#endif //PIPE_TRANSMISSION_IRGBDRECEIVER_H

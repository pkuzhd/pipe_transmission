//
// Created by zhanghaodan on 2022/7/11.
//

#ifndef PIPE_TRANSMISSION_RGBDRECEIVER_H
#define PIPE_TRANSMISSION_RGBDRECEIVER_H

#include <string>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "IRGBDReceiver.h"

class RGBDReceiver : public IRGBDReceiver {
public:
    int32_t fd;
    std::queue<RGBDData *> queue;
    std::mutex m;
    std::condition_variable cv;
    
    int bufSize;
    int queueSize;

    RGBDReceiver();

    RGBDReceiver(int bufSize, int queueSize);

    ~RGBDReceiver();

    void addData(RGBDData *data);

    int open(std::string filename) override;

    int close() override;

    bool isFileExists_stat(std::string &name);

    RGBDData *getSingleFrame();

    RGBDData *getData() override;

    int getBufferSize() override;
};


#endif //PIPE_TRANSMISSION_RGBDRECEIVER_H

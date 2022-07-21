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


struct RGBDData {
    int n;
    int *w, *h;
    int *w_crop, *h_crop, *x, *y;
    char *imgs;
    char *depths;
    char *masks;

    char *getImage(int i);
    char *getDepth(int i);
    char *getMask(int i);
};

class RGBDReceiver {
public:
    int32_t fd;
    std::queue<RGBDData *> queue;
    std::mutex m;
    RGBDReceiver();
    ~RGBDReceiver();
    void addData(RGBDData *data);
    int open(std::string filename);
    int close();
    bool isFileExists_stat(std::string& name);

    RGBDData *getSingleFrame();
    RGBDData *getData();
};


#endif //PIPE_TRANSMISSION_RGBDRECEIVER_H

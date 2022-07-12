//
// Created by zhanghaodan on 2022/7/11.
//

#ifndef PIPE_TRANSMISSION_RGBDRECEIVER_H
#define PIPE_TRANSMISSION_RGBDRECEIVER_H

#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <json.h>

#define ARRAYSIZE 1024

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

    RGBDReceiver();
    ~RGBDReceiver();

    int open(std::string filename);
    int close();
    int IsFileExist(const char* path);

    RGBDData *getData();
};


#endif //PIPE_TRANSMISSION_RGBDRECEIVER_H

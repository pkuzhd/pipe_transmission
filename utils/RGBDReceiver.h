//
// Created by zhanghaodan on 2022/7/11.
//

#ifndef PIPE_TRANSMISSION_RGBDRECEIVER_H
#define PIPE_TRANSMISSION_RGBDRECEIVER_H

#include <string>


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
    RGBDReceiver();
    ~RGBDReceiver();

    int open(std::string filename);
    int close();

    RGBDData *getData();
};


#endif //PIPE_TRANSMISSION_RGBDRECEIVER_H

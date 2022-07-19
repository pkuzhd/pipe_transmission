//
// Created by zhanghaodan on 2022/7/19.
//

#ifndef PIPE_TRANSMISSION_IRGBDRECEIVER_H
#define PIPE_TRANSMISSION_IRGBDRECEIVER_H

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

class IRGBDReceiver {
public:

    virtual ~IRGBDReceiver() = 0;

    virtual int open(std::string filename) = 0;

    virtual int close() = 0;

    virtual RGBDData *getData() = 0;
};


#endif //PIPE_TRANSMISSION_IRGBDRECEIVER_H

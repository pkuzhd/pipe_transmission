//
// Created by zhanghaodan on 2022/7/19.
//

#ifndef PIPE_TRANSMISSION_RGBDDATA_H
#define PIPE_TRANSMISSION_RGBDDATA_H


class RGBDData {
public:
    int n;
    int *w, *h;
    int *w_crop, *h_crop, *x, *y;
    char *imgs;
    char *depths;
    char *masks;

    RGBDData();

    ~RGBDData();

    char *getImage(int i);

    char *getDepth(int i);

    char *getMask(int i);
};


#endif //PIPE_TRANSMISSION_RGBDDATA_H

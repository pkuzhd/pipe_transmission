//
// Created by zhanghaodan on 2022/7/11.
//

#ifndef PIPE_TRANSMISSION_IMAGESENDER_H
#define PIPE_TRANSMISSION_IMAGESENDER_H

#include <string>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <vector>

struct ImageData {
    int n;
    int *w, *h;
    char *imgs;
};

class ImageSender {
public:
    int32_t fd;

    ImageSender();

    ~ImageSender();

    int open(std::string filename);

    int close();

    int sendData(ImageData *data);

    bool isFileExists_stat(std::string &name);
};


#endif //PIPE_TRANSMISSION_IMAGESENDER_H

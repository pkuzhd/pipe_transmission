//
// Created by zhanghaodan on 2022/7/11.
//

#ifndef PIPE_TRANSMISSION_IMAGESENDER_H
#define PIPE_TRANSMISSION_IMAGESENDER_H

#include <string>

struct ImageData {
    int n;
    int *w, *h;
    char *imgs;
};

class ImageSender {
public:
    ImageSender();

    ~ImageSender();

    int open(std::string filename);

    int close();

    int sendData(ImageData *data);
};


#endif //PIPE_TRANSMISSION_IMAGESENDER_H

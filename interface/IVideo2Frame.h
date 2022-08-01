//
// Created by zhanghaodan on 2022/8/1.
//

#ifndef PIPE_TRANSMISSION_IVIDEO2FRAME_H
#define PIPE_TRANSMISSION_IVIDEO2FRAME_H

#include <inttypes.h>
#include <vector>

enum Status {
    SUCCESS,
    FAILED
};

struct Frame {
    uint32_t width, height, id;
    double pts;
    char *images;
};

class IVideo2Frame {
    virtual Frame *getFrame() = 0;
};


class IFrameSender {
    virtual Status sendFrame(std::vector<Frame *> &frames) = 0;
};

struct RGBDMData {
    uint32_t width, height, n;
    double pts;
    char *images;
    char *depths;
    char *masks;
};

class IRGBDMDataReceiver {
    virtual RGBDMData *recvRGBDMData() = 0;
};

#endif //PIPE_TRANSMISSION_IVIDEO2FRAME_H

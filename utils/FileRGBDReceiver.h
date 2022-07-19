//
// Created by zhanghaodan on 2022/7/19.
//

#ifndef PIPE_TRANSMISSION_FILERGBDRECEIVER_H
#define PIPE_TRANSMISSION_FILERGBDRECEIVER_H

#include "IRGBDReceiver.h"

#include <queue>
#include <thread>
#include <mutex>

class FileRGBDReceiver : public IRGBDReceiver {
public:
    FileRGBDReceiver();

    ~FileRGBDReceiver();

    int open(std::string filename) override;

    int close() override;

    RGBDData *getData() override;

    int getBufferSize() override;

public:
    int current_idx = 1;
    std::string path;
    std::queue<RGBDData *> buffer;
    std::mutex m;
    std::thread *t;

    RGBDData *_getData();
};


#endif //PIPE_TRANSMISSION_FILERGBDRECEIVER_H

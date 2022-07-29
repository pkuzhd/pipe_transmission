//
// Created by zhanghaodan on 2022/7/11.
//

#include "RGBDReceiver.h"
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>

// TODO: P1 use unique_lock
// TODO: P1 malloc/free -> new/delete
// TODO: P1 remove log
// TODO: P1 format code

// TODO: P3 exit
void readdata_thread(RGBDReceiver *R) {
    while (1) {
        RGBDData *rgbdData = R->getSingleFrame();
        while (!rgbdData) {
            rgbdData = R->getSingleFrame();
        }
        R->addData(rgbdData);
    }
}

// TODO: P2 add condition variable
void RGBDReceiver::addData(RGBDData *data) {

    while (1) {
        bool flag;
        m.lock();
        if (queue.size() > 10) { // TODO: P1 add buffer size
            flag = 0;
        } else {
            queue.push(data);
            flag = 1;
        }
        m.unlock();
        if (flag == 1) {
            break;
        }
    }
}

RGBDData *RGBDReceiver::getData() {
    m.lock();
    if (queue.size() == 0) {
        m.unlock();
        return nullptr;
    }
    RGBDData *rgbdData = queue.front();
    queue.pop();
    m.unlock();
    return rgbdData;

}

// TODO: P1 add pipe size
RGBDData *RGBDReceiver::getSingleFrame() {
    char * readBuf = new char[bufSize]; // TODO: P1 use new/delete cc
    memset(readBuf, '\0', bufSize);
    RGBDData *rgbdData = new RGBDData;
    int tmp;

    if ((tmp = read(fd, readBuf, 121)) <= 0) {
        std::cout << "read error\n";
        delete []readBuf;
        delete rgbdData;
        return nullptr;
    } else {

        int cur = 0;

        rgbdData->n = *(int8_t*) (readBuf + cur);
        cur += sizeof(unsigned char);

        rgbdData->w = new int[rgbdData->n];
        rgbdData->h = new int[rgbdData->n];
        rgbdData->x = new int[rgbdData->n];
        rgbdData->y = new int[rgbdData->n];
        rgbdData->w_crop = new int[rgbdData->n];
        rgbdData->h_crop = new int[rgbdData->n];


        for (int i = 0; i < rgbdData->n; i++) {
            // rgbdData->w[i] = *(int*) (readBuf + cur);
            rgbdData->h[i] = *(int*) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->w[i] = *(int*) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->w_crop[i] = *(int*) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->h_crop[i] = *(int*) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->x[i] = *(int*) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->y[i] = *(int*) (readBuf + cur);
            cur += sizeof(unsigned int);

            std::cout << "now w[i] is: " << rgbdData->w[i] << std::endl;
            std::cout << "now h[i] is: " << rgbdData->h[i] << std::endl;
            std::cout << "now x[i] is: " << rgbdData->x[i] << std::endl;
            std::cout << "now y[i] is: " << rgbdData->y[i] << std::endl;
            std::cout << "now w_crop[i] is: " << rgbdData->w_crop[i] << std::endl;
            std::cout << "now h_crop[i] is: " << rgbdData->h_crop[i] << std::endl;

        }

        int total_imgs_len = 0;
        for (int i = 0; i < rgbdData->n; i++) {
            total_imgs_len += rgbdData->w[i] * rgbdData->h[i] * 3;
        }
        int total_depths_len = 0;
        for (int i = 0; i < rgbdData->n; i++) {
            total_depths_len += rgbdData->w_crop[i] * rgbdData->h_crop[i] * 4;
        }

        int total_masks_len = 0;
        for (int i = 0; i < rgbdData->n; i++) {
            total_masks_len += rgbdData->w_crop[i] * rgbdData->h_crop[i];
        }
        
        // std::cout << total_imgs_len << std::endl;
        

        rgbdData->imgs = new char[total_imgs_len];
        rgbdData->depths = new char[total_depths_len];
        rgbdData->masks = new char[total_masks_len];

        // add the imgs
        memset(readBuf, '\0', bufSize);
        int imgs_len = 0;

        int i = 0;
        while (imgs_len + bufSize < total_imgs_len) {
            int len = read(fd, readBuf, bufSize);
            memcpy(rgbdData->imgs + imgs_len, readBuf, len);
            imgs_len += len;
            memset(readBuf, '\0', bufSize);
        }


        // add the remain total - imgs_len part
        memset(readBuf, '\0', bufSize);
        int len = read(fd, readBuf, total_imgs_len - imgs_len);

        memcpy(rgbdData->imgs + imgs_len, readBuf, total_imgs_len - imgs_len);
        imgs_len += len;


        // add the depths part
        memset(readBuf, '\0', bufSize);
        int depths_len = 0;

        i = 0;
        while (depths_len + bufSize < total_depths_len) {
            int len = read(fd, readBuf, bufSize);
            memcpy(rgbdData->depths + depths_len, readBuf, len);
            depths_len += len;
            memset(readBuf, '\0', bufSize);
        }


        // add the remain total - depths_len part
        memset(readBuf, '\0', bufSize);
        len = read(fd, readBuf, total_depths_len - depths_len);
        memcpy(rgbdData->depths + depths_len, readBuf, total_depths_len - depths_len);
        depths_len += len;
        
        

        // add the masks part
        memset(readBuf, '\0', bufSize);
        int masks_len = 0;
        i = 0;
        while (masks_len + bufSize < total_masks_len) {
            int len = read(fd, readBuf, bufSize);
            memcpy(rgbdData->masks + masks_len, readBuf, len);
            masks_len += len;
            memset(readBuf, '\0', bufSize);
        }

        // add the remain total - masks_len part
        memset(readBuf, '\0', bufSize);
        len = read(fd, readBuf, total_masks_len - masks_len);
        memcpy(rgbdData->masks + masks_len, readBuf, total_masks_len - masks_len);
        masks_len += len;
        
        delete []readBuf;
        return rgbdData;
    }
    return nullptr;
}

RGBDReceiver::~RGBDReceiver() {

}

RGBDReceiver::RGBDReceiver() {
    bufSize = 1048576;
    queueSize = 64;
}
RGBDReceiver::RGBDReceiver(int bufSize, int queueSize) {
    this->bufSize = bufSize;
    this->queueSize = queueSize;
}

int RGBDReceiver::open(std::string filename) {
    if (!isFileExists_stat(filename)) {
        std::cout << filename << std::endl;
        int32_t ret = mkfifo(filename.c_str(), S_IFIFO | 0666);
        if (ret == -1) {
            std::cout << errno << std::endl;
            std::cout << "Make fifo error\n";
            return -1;
        }
    }
    fd = ::open(filename.c_str(), O_RDONLY);
    fcntl(fd, 1031, 1048576);
    if (fd < 0) {
        return fd;
    }
    std::thread th(readdata_thread, this);
    th.detach();
    return 0;
}

int RGBDReceiver::close() {
    int ret = ::close(fd);
    return ret;
}

bool RGBDReceiver::isFileExists_stat(std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

int RGBDReceiver::getBufferSize() {
    int size = 0;
    m.lock();
    size = queue.size();
    m.unlock();
    return size;
}

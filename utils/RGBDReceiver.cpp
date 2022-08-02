//
// Created by zhanghaodan on 2022/7/11.
//

#include "RGBDReceiver.h"
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>

// TODO: P1 use unique_lock (done)
// TODO: P1 malloc/free -> new/delete (done)
// TODO: P1 remove log (done)
// TODO: P1 format code (done)

// TODO: P3 exit (done)
void readdata_thread(RGBDReceiver *R) {
    while (1) {
        {
            std::unique_lock<std::mutex> guard(R->m);
            if(R->is_exit){
                break;
            }
        }
        RGBDData *rgbdData = R->getSingleFrame();
        while (!rgbdData) {
            rgbdData = R->getSingleFrame();
        }
        R->addData(rgbdData);
    }
}

// TODO: P2 add condition variable (done)
void RGBDReceiver::addData(RGBDData *data) {

    std::unique_lock<std::mutex> guard(m);
    while (queue.size() > queueSize) { // TODO: P1 add buffer size (done)
        not_full.wait(guard);
    }
    queue.push(data);
}

RGBDData *RGBDReceiver::getData() {
    std::unique_lock<std::mutex> guard(m);
    not_full.notify_one();
    if (queue.size() == 0) {
        return nullptr;
    }
    RGBDData *rgbdData = queue.front();
    queue.pop();
    return rgbdData;

}

// TODO: P1 add pipe size (done)
RGBDData *RGBDReceiver::getSingleFrame() {
    char *readBuf = new char[bufSize]; // TODO: P1 use new/delete (done)
    memset(readBuf, '\0', bufSize);
    RGBDData *rgbdData = new RGBDData;
    int tmp;

    if ((tmp = read(fd, readBuf, 4)) <= 0) {
        std::cout << "read error\n";
        delete[]readBuf;
        delete rgbdData;
        return nullptr;
    } else {

        int cur = 0;

        rgbdData->n = *(int *) (readBuf + cur);
        if(rgbdData->n == -1){
            delete[]readBuf;
            delete rgbdData;
            return nullptr;
        }

        cur += sizeof(unsigned int);

        tmp = read(fd, readBuf + cur, 120);

        std::cout << "now n is: " << rgbdData->n << std::endl;

        rgbdData->w = new int[rgbdData->n];
        rgbdData->h = new int[rgbdData->n];
        rgbdData->x = new int[rgbdData->n];
        rgbdData->y = new int[rgbdData->n];
        rgbdData->w_crop = new int[rgbdData->n];
        rgbdData->h_crop = new int[rgbdData->n];


        for (int i = 0; i < rgbdData->n; i++) {
            // rgbdData->w[i] = *(int*) (readBuf + cur);
            rgbdData->h[i] = *(int *) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->w[i] = *(int *) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->w_crop[i] = *(int *) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->h_crop[i] = *(int *) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->x[i] = *(int *) (readBuf + cur);
            cur += sizeof(unsigned int);
            rgbdData->y[i] = *(int *) (readBuf + cur);
            cur += sizeof(unsigned int);

            std::cout << "now w[i] is: " << rgbdData->w[i] << std::endl;
            std::cout << "now h[i] is: " << rgbdData->h[i] << std::endl;
            std::cout << "now x[i] is: " << rgbdData->x[i] << std::endl;
            std::cout << "now y[i] is: " << rgbdData->y[i] << std::endl;
            std::cout << "now w_crop[i] is: " << rgbdData->w_crop[i] << std::endl;
            std::cout << "now h_crop[i] is: " << rgbdData->h_crop[i] << std::endl;

        }

        ssize_t total_imgs_len = 0;
        for (int i = 0; i < rgbdData->n; i++) {
            total_imgs_len += rgbdData->w[i] * rgbdData->h[i] * 3;
        }
        ssize_t total_depths_len = 0;
        for (int i = 0; i < rgbdData->n; i++) {
            total_depths_len += rgbdData->w_crop[i] * rgbdData->h_crop[i] * 4;
        }

        ssize_t total_masks_len = 0;
        for (int i = 0; i < rgbdData->n; i++) {
            total_masks_len += rgbdData->w_crop[i] * rgbdData->h_crop[i];
        }

        // std::cout << total_imgs_len << std::endl;


        rgbdData->imgs = new char[total_imgs_len];
        rgbdData->depths = new char[total_depths_len];
        rgbdData->masks = new char[total_masks_len];

        // add the imgs
        ssize_t imgs_len = 0;
        while (imgs_len + bufSize < total_imgs_len) {
            ssize_t len = read(fd, rgbdData->imgs + imgs_len, bufSize);
            imgs_len += len;
        }


        // add the remain total - imgs_len part
        ssize_t len = read(fd, rgbdData->imgs + imgs_len, total_imgs_len - imgs_len);
        imgs_len += len;


        // add the depths part
        ssize_t depths_len = 0;


        while (depths_len + bufSize < total_depths_len) {
            ssize_t len = read(fd, rgbdData->depths + depths_len, bufSize);
            depths_len += len;
        }


        // add the remain total - depths_len part
        len = read(fd, rgbdData->depths + depths_len, total_depths_len - depths_len);
        depths_len += len;



        // add the masks part
        ssize_t masks_len = 0;
        while (masks_len + bufSize < total_masks_len) {
            int len = read(fd, rgbdData->masks + masks_len, bufSize);
            masks_len += len;
        }

        // add the remain total - masks_len part

        len = read(fd, rgbdData->masks + masks_len, total_masks_len - masks_len);
        masks_len += len;

        delete[]readBuf;
        return rgbdData;
    }
    return nullptr;
}

RGBDReceiver::~RGBDReceiver() {
    {
        std::unique_lock<std::mutex> guard(m);
        is_exit = 1;
    }
    th.join();
}

RGBDReceiver::RGBDReceiver(): bufSize(1048576), queueSize(64), is_exit(0){

}

RGBDReceiver::RGBDReceiver(int bufSize, int queueSize): bufSize(bufSize), queueSize(queueSize){

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
    this->th = std::move(th);
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

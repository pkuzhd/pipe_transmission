//
// Created by zhanghaodan on 2022/7/11.
//

#include "RGBDReceiver.h"
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>

void readdata_thread(RGBDReceiver *R) {
    while (1) {
        RGBDData *rgbdData = R->getSingleFrame();
        while (!rgbdData) {
            rgbdData = R->getSingleFrame();
        }
        R->addData(rgbdData);
    }
}

void RGBDReceiver::addData(RGBDData *data) {

    while (1) {
        bool flag;
        m.lock();
        if (queue.size() > 10) {
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
    std::cout << "now queue size is:" << queue.size() << std::endl;
    RGBDData *rgbdData = queue.front();
    queue.pop();
    m.unlock();
    return rgbdData;

}

RGBDData *RGBDReceiver::getSingleFrame() {
    const int32_t bufSize = 1048576;
    char readBuf[bufSize];
    memset(readBuf, '\0', bufSize);
    RGBDData *rgbdData = (RGBDData *) malloc(sizeof(RGBDData));
    int tmp;

    if ((tmp = read(fd, readBuf, 121)) <= 0) {
        std::cout << "read error\n";
        free(rgbdData);
        return nullptr;
    } else {
        std::cout << "121 " << tmp << std::endl;
        int cur = 0;

        uint8_t N;
        memcpy(&N, readBuf, sizeof(unsigned char));
        rgbdData->n = (int) N;
        std::cout << "now length is: " << rgbdData->n << std::endl;

        cur += sizeof(unsigned char);

        printf("11111");

        rgbdData->w = (int *) malloc(sizeof(int) * rgbdData->n);
        rgbdData->h = (int *) malloc(sizeof(int) * rgbdData->n);
        rgbdData->x = (int *) malloc(sizeof(int) * rgbdData->n);
        rgbdData->y = (int *) malloc(sizeof(int) * rgbdData->n);
        rgbdData->w_crop = (int *) malloc(sizeof(int) * rgbdData->n);
        rgbdData->h_crop = (int *) malloc(sizeof(int) * rgbdData->n);

        printf("11112");


        for (int i = 0; i < rgbdData->n; i++) {
            uint32_t h;
            memcpy(&h, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);

            uint32_t w;
            memcpy(&w, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);

            uint32_t w_crop;
            memcpy(&w_crop, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);

            uint32_t h_crop;
            memcpy(&h_crop, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);


            uint32_t x;
            memcpy(&x, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);


            uint32_t y;
            memcpy(&y, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);

            rgbdData->w[i] = (int) w;
            rgbdData->h[i] = (int) h;
            rgbdData->x[i] = (int) x;
            rgbdData->y[i] = (int) y;
            rgbdData->w_crop[i] = (int) w_crop;
            rgbdData->h_crop[i] = (int) h_crop;

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
        std::cout << total_imgs_len << std::endl;
        int total_depths_len = 0;
        for (int i = 0; i < rgbdData->n; i++) {
            total_depths_len += rgbdData->w_crop[i] * rgbdData->h_crop[i] * 4;
        }
        std::cout << total_depths_len << std::endl;
        int total_masks_len = 0;
        for (int i = 0; i < rgbdData->n; i++) {
            total_masks_len += rgbdData->w_crop[i] * rgbdData->h_crop[i];
        }
        std::cout << total_masks_len << std::endl;

        std::cout << "3333" << std::endl;

        rgbdData->imgs = (char *) malloc(total_imgs_len);
        rgbdData->depths = (char *) malloc(total_depths_len);
        rgbdData->masks = (char *) malloc(total_masks_len);

        std::cout << "3334" << std::endl;

        // add the imgs
        memset(readBuf, '\0', bufSize);
        int imgs_len = 0;
        std::cout << "3335" << std::endl;
        int i = 0;
        while (imgs_len + bufSize < total_imgs_len) {
            int len = read(fd, readBuf, bufSize);
            memcpy(rgbdData->imgs + imgs_len, readBuf, len);
            imgs_len += len;
            memset(readBuf, '\0', bufSize);
        }
        std::cout << "3336" << std::endl;

        // add the remain total - imgs_len part
        memset(readBuf, '\0', bufSize);
        int len = read(fd, readBuf, total_imgs_len - imgs_len);
        std::cout << len << std::endl;
        memcpy(rgbdData->imgs + imgs_len, readBuf, total_imgs_len - imgs_len);
        imgs_len += len;

        std::cout << imgs_len << " " << total_imgs_len << std::endl;

        // add the depths part
        memset(readBuf, '\0', bufSize);
        int depths_len = 0;
        std::cout << "3337" << std::endl;
        i = 0;
        while (depths_len + bufSize < total_depths_len) {
            int len = read(fd, readBuf, bufSize);
            memcpy(rgbdData->depths + depths_len, readBuf, len);
            depths_len += len;
            memset(readBuf, '\0', bufSize);
        }
        std::cout << "3338" << std::endl;

        std::cout << depths_len << " " << total_depths_len << std::endl;

        // add the remain total - depths_len part
        memset(readBuf, '\0', bufSize);
        len = read(fd, readBuf, total_depths_len - depths_len);
        std::cout << len << std::endl;
        memcpy(rgbdData->depths + depths_len, readBuf, total_depths_len - depths_len);
        depths_len += len;

        std::cout << depths_len << " " << total_depths_len << std::endl;

        // add the masks part
        memset(readBuf, '\0', bufSize);
        int masks_len = 0;
        std::cout << "3339" << std::endl;
        i = 0;
        while (masks_len + bufSize < total_masks_len) {
            int len = read(fd, readBuf, bufSize);
            memcpy(rgbdData->masks + masks_len, readBuf, len);
            masks_len += len;
            memset(readBuf, '\0', bufSize);
        }
        std::cout << "33310" << std::endl;

        std::cout << masks_len << " " << total_masks_len << std::endl;

        // add the remain total - masks_len part
        memset(readBuf, '\0', bufSize);
        len = read(fd, readBuf, total_masks_len - masks_len);
        std::cout << len << std::endl;
        memcpy(rgbdData->masks + masks_len, readBuf, total_masks_len - masks_len);
        masks_len += len;

        std::cout << masks_len << " " << total_masks_len << std::endl;


        return rgbdData;
    }
    return nullptr;
}

RGBDReceiver::~RGBDReceiver() {

}

RGBDReceiver::RGBDReceiver() {

}

int RGBDReceiver::open(std::string filename) {
    if (isFileExists_stat(filename)) {
        std::cout << filename << std::endl;
//        remove(filename.c_str());
    } else {
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

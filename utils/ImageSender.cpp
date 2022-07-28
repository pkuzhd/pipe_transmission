//
// Created by zhanghaodan on 2022/7/11.
//

#include "ImageSender.h"
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctime>

#define BUFSIZE 1000000000

ImageSender::ImageSender() {

}

ImageSender::~ImageSender() {

}

int ImageSender::open(std::string filename) {
    if (!isFileExists_stat(filename)) {
        std::cout << filename << std::endl;
        int32_t ret = mkfifo(filename.c_str(), S_IFIFO | 0666);
        if (ret == -1) {
            std::cout << errno << std::endl;
            std::cout << "Make fifo error\n";
            return -1;
        }
    }

    fd = ::open(filename.c_str(), O_WRONLY);
    if (fd < 0) {
        return fd;
    }
    return 0;
}

int ImageSender::close() {
    int ret = ::close(fd);
    return ret;
}

int ImageSender::sendData(ImageData *data) {
    char *bytes = (char *) malloc(BUFSIZE);
    int cur = 0;
    int cur_img = 0;
    int n = data->n;
    memcpy(bytes, &n, 4);
    cur += 4;
    for (int i = 0; i < n; i++) {
        memcpy(bytes + cur, &(data->w[i]), 4);
        cur += 4;
        memcpy(bytes + cur, &(data->h[i]), 4);
        cur += 4;
        memcpy(bytes + cur, data->imgs + cur_img, data->w[i] * data->h[i] * 3);
        cur += data->w[i] * data->h[i] * 3;
        cur_img += data->w[i] * data->h[i] * 3;
        //std::cout << "now cur_img is:" << cur_img << std::endl;
    }
    //std::cout << "now cur is : " << cur << std::endl;

    int len = write(fd, bytes, cur);

    free(bytes);
    std::cout << "total bytes sent is: " << len << std::endl;
    return len;

}

bool ImageSender::isFileExists_stat(std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

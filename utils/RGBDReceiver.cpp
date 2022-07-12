//
// Created by zhanghaodan on 2022/7/11.
//

#include "RGBDReceiver.h"

char *RGBDData::getImage(int i) {
    int bias = 0;
    for(int j = 0 ; j < i ; j++){
        bias += w[j] * h[j] * 3;
    }
    return imgs + bias;
}

char *RGBDData::getDepth(int i) {
    int bias = 0;
    for(int j = 0 ; j < i ; j++){
        bias += w_crop[j] * h_crop[j] * 4;
    }
    return imgs + bias;
}

char *RGBDData::getMask(int i) {
    int bias = 0;
    for(int j = 0 ; j < i ; j++){
        bias += w_crop[j] * h_crop[j];
    }
    return imgs + bias;
}

RGBDData *RGBDReceiver::getData() {
    const int32_t bufSize = 82;
    char readBuf[bufSize];
    memset(readBuf, '\0', bufSize);
    RGBDData * rgbdData = (RGBDData*) malloc(sizeof(RGBDData));

    if (read(fd, readBuf, bufSize) < 0) {
        std::cout << "read error\n";
        return nullptr;
    }
    else {
        int cur = 0;

        uint8_t N;
        memcpy(&N, readBuf, sizeof(unsigned char));
        rgbdData->n = (int)N;
        std::cout << "now length is: " << rgbdData->n << std::endl;

        cur += sizeof(unsigned char);

        rgbdData->w = (int*) malloc(sizeof(int) * N);
        rgbdData->h = (int*) malloc(sizeof(int) * N);
        rgbdData->x = (int*) malloc(sizeof(int) * N);
        rgbdData->y = (int*) malloc(sizeof(int) * N);
        rgbdData->w_crop = (int*) malloc(sizeof(int) * N);
        rgbdData->h_crop = (int*) malloc(sizeof(int) * N);

        for(int i = 0 ; i < N ; i++){
            uint32_t w;
            memcpy(&w, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);

            uint32_t h;
            memcpy(&h, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);

            uint32_t x;
            memcpy(&x, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);

            uint32_t y;
            memcpy(&y, readBuf + cur, sizeof(unsigned int));

            cur += sizeof(unsigned int);

            rgbdData->w[i] = (int)w;
            rgbdData->h[i] = (int)h;
            rgbdData->x[i] = (int)x;
            rgbdData->y[i] = (int)y;
            rgbdData->w_crop[i] = (int)w;
            rgbdData->h_crop[i] = (int)h;




        }


        return rgbdData;
    }
    return nullptr;
}

RGBDReceiver::~RGBDReceiver() {

}

RGBDReceiver::RGBDReceiver() {

}

int RGBDReceiver::open(std::string filename) {
    if(isFileExists_stat(filename)){
        std::cout << filename << std::endl;
        remove(filename.c_str());
    }
    int32_t ret = mkfifo(filename.c_str(), S_IFIFO | 0666);
    if (ret == -1){
        std::cout << errno << std::endl;
        std::cout << "Make fifo error\n";
        return -1;
    }
    fd = ::open(filename.c_str(), O_RDONLY);
    if(fd < 0){
        return fd;
    }
    return 0;
}

int RGBDReceiver::close() {
    int ret = ::close(fd);
    return ret;
}
bool RGBDReceiver::isFileExists_stat(std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

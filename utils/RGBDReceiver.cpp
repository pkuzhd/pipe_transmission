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
        free(rgbdData);
        return nullptr;
    }
    else {
        int cur = 0;

        uint8_t N;
        memcpy(&N, readBuf, sizeof(unsigned char));
        rgbdData->n = (int)N;
        std::cout << "now length is: " << rgbdData->n << std::endl;

        cur += sizeof(unsigned char);

        rgbdData->w = (int*) malloc(sizeof(int) * rgbdData->n);
        rgbdData->h = (int*) malloc(sizeof(int) * rgbdData->n);
        rgbdData->x = (int*) malloc(sizeof(int) * rgbdData->n);
        rgbdData->y = (int*) malloc(sizeof(int) * rgbdData->n);
        rgbdData->w_crop = (int*) malloc(sizeof(int) * rgbdData->n);
        rgbdData->h_crop = (int*) malloc(sizeof(int) * rgbdData->n);


        for(int i = 0 ; i < rgbdData->n ; i++){
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

        int total_imgs_len = 0;
        for(int i = 0 ; i < N ; i++){
            total_imgs_len += rgbdData->w[i] * rgbdData->h[i] * 3;
        }
        int total_depths_len = 0;
        for(int i = 0 ; i < N ; i++){
            total_depths_len += rgbdData->w_crop[i] * rgbdData->h_crop[i] * 4;
        }
        int total_masks_len = 0;
        for(int i = 0 ; i < N ; i++){
            total_masks_len += rgbdData->w_crop[i] * rgbdData->h_crop[i];
        }

        rgbdData->imgs = (char*) malloc(sizeof(total_imgs_len));
        rgbdData->depths = (char*) malloc(sizeof(total_depths_len));
        rgbdData->masks = (char*) malloc(sizeof(total_masks_len));

        memset(readBuf, '\0', bufSize);
        int imgs_len = 0;
        while(imgs_len < total_imgs_len){
            int len = read(fd, readBuf, bufSize);
            memcpy(rgbdData->imgs + imgs_len, readBuf, bufSize);
            imgs_len += len;
            memset(readBuf, '\0', bufSize);
        }
        if(imgs_len > total_imgs_len){
            char temp[imgs_len - total_imgs_len + 3];
            memcpy(temp, rgbdData->imgs + total_imgs_len, imgs_len - total_imgs_len);
            temp[imgs_len - total_imgs_len] = '\0';
            memset(temp, '\0', sizeof(temp));
            memcpy(rgbdData->depths, temp, imgs_len - total_imgs_len);
        }
        std::cout << imgs_len << " " << total_imgs_len << std::endl;


        memset(readBuf, '\0', bufSize);
        int depths_len = imgs_len - total_imgs_len;
        while(depths_len < total_depths_len){
            int len = read(fd, readBuf, bufSize);
            memcpy(rgbdData->depths + depths_len, readBuf, bufSize);
            depths_len += len;
            memset(readBuf, '\0', bufSize);
        }
        if(depths_len > total_depths_len){
            char temp[depths_len - total_depths_len + 3];
            memcpy(temp, rgbdData->depths + total_depths_len, depths_len - total_depths_len);
            temp[depths_len - total_depths_len] = '\0';
            memset(temp, '\0', sizeof(temp));
            memcpy(rgbdData->masks, temp, depths_len - total_depths_len);
        }
        std::cout << depths_len  << " " << total_depths_len << std::endl;


        memset(readBuf, '\0', bufSize);
        int masks_len = depths_len - total_depths_len;
        while(masks_len < total_masks_len){
            int len = read(fd, readBuf, bufSize);
            memcpy(rgbdData->masks + masks_len, readBuf, bufSize);
            masks_len += len;
            memset(readBuf, '\0', bufSize);
        }

        std::cout << masks_len  << " " << total_masks_len << std::endl;



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

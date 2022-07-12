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
        bias += w[j] * h[j];
    }
    return imgs + bias;
}

RGBDData *RGBDReceiver::getData() {
    const int32_t bufSize = 1024;
    char readBuf[bufSize];
    if (read(fd, readBuf, bufSize) < 0) {
        std::cout << "read error\n";
        return NULL;
    }
    else {
        Json::Reader reader;
        Json::Value root;
        RGBDData * rgbdData = new RGBDData;

        rgbdData->imgs = (char*) malloc(sizeof(ARRAYSIZE));
        rgbdData->depths = (char*) malloc(sizeof(ARRAYSIZE));
        rgbdData->masks = (char*) malloc(sizeof(ARRAYSIZE));


        if (reader.parse(readBuf, root)){
            int n = root["N"].asInt();
            std::string imgs = root["imgs"].asString();
            std::string depths = root["depths"].asString();
            std::string masks = root["masks"].asString();
            rgbdData->n = n;
            strcpy(rgbdData->imgs, imgs.c_str());
            strcpy(rgbdData->depths, depths.c_str());
            strcpy(rgbdData->masks, masks.c_str());
            for(int i = 0 ; i < 5 ; i++){
                std::string crop = root["crops"][i].asString();
                cout << crop << endl;
            }

        }

        std::cout << "recv msg: ";
        std::cout << readBuf << '\n';
        return rgbdData;
    }
    return nullptr;
}

RGBDReceiver::~RGBDReceiver() {

}

RGBDReceiver::RGBDReceiver() {

}

int RGBDReceiver::open(std::string filename) {
    if(IsFileExist(filename.c_str())){
        remove(filename.c_str());
    }
    int32_t ret = mkfifo(filename.c_str(), S_IFIFO | 0666);
    if (ret == -1){
        std::cout << "Make fifo error\n";
        return -1;
    }
    fd = ::open(filename.c_str(), O_RDONLY);
    return 0;
}

int RGBDReceiver::close() {
    ::close(fd);
    return 0;
}
int IsFileExist(const char* path){
    return !access(path, F_OK);
}

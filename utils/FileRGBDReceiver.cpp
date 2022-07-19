//
// Created by zhanghaodan on 2022/7/19.
//

#include "FileRGBDReceiver.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

FileRGBDReceiver::FileRGBDReceiver() {

}

FileRGBDReceiver::~FileRGBDReceiver() {

}

int FileRGBDReceiver::open(std::string filename) {
    path = filename;
    return 0;
}

int FileRGBDReceiver::close() {
    return 0;
}

RGBDData *FileRGBDReceiver::getData() {
    int crop[5][2] = {
            {800, 150},
            {800, 150},
            {750, 180},
            {550, 180},
            {500, 180}
    };
    RGBDData *data = new RGBDData;
    data->n = 5;
    data->w = new int[data->n];
    data->h = new int[data->n];
    data->w_crop = new int[data->n];
    data->h_crop = new int[data->n];
    data->x = new int[data->n];
    data->y = new int[data->n];
    for (int i = 0; i < 5; ++i) {
        data->w[i] = 1920;
        data->h[i] = 1080;
        data->w_crop[i] = 800;
        data->h_crop[i] = 900;
        data->x[i] = crop[i][0];
        data->y[i] = crop[i][1];
    }
    data->imgs = new char[1920 * 1080 * 3 * 5];
    data->depths = new char[data->w_crop[0] * data->h_crop[0] * 4 * 3 * 5];
    data->masks = new char[data->w_crop[0] * data->h_crop[0] * 3 * 5];
    for (int i = 0; i < 5; ++i) {
        cv::Mat img = cv::imread(path + "video/" + to_string(i + 1) + "-" + to_string(current_idx * 5 + 96) + ".png");
        memcpy(data->getImage(i), img.data, 1920 * 1080 * 3);
    }
    for (int i = 0; i < 5; ++i) {
        cv::Mat mask = cv::imread(path + "mask/" + to_string(i + 1) + "-" + to_string(current_idx * 5 + 96) + ".png",
                                  cv::IMREAD_GRAYSCALE);
        memcpy(data->getMask(i), mask.data, data->w_crop[0] * data->h_crop[0]);
    }
    for (int i = 0; i < 5; ++i) {
        FILE *f = fopen((path + "depth/" + to_string(i + 1) + "-" + to_string(current_idx * 5 + 96) + ".depth").c_str(), "rb");
        fread(data->getDepth(i), data->w_crop[0] * data->h_crop[0] * 4, 1, f);
        fclose(f);
    }

    ++current_idx;
    return data;
}

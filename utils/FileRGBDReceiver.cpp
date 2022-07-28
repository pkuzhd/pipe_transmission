//
// Created by zhanghaodan on 2022/7/19.
//

#include "FileRGBDReceiver.h"

#include "ThreadPool.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

void read_thread(FileRGBDReceiver *receiver) {
    while (true) {
        RGBDData *data = receiver->_getData();
        int size = 0;
        {
            unique_lock<mutex> lock(receiver->m);
            receiver->buffer.push(data);
            size = receiver->buffer.size();
            while (size > 100) {
//                cout << "sleep" << endl;
                receiver->not_full.wait(lock);
//                cout << "end sleep" << endl;
                size = receiver->buffer.size();
//                cout << size << endl;
            }
        }
    }
}

FileRGBDReceiver::FileRGBDReceiver() {

}

FileRGBDReceiver::~FileRGBDReceiver() {

}

int FileRGBDReceiver::open(std::string filename) {
    path = filename;
    t = new thread(read_thread, this);
    return 0;
}

int FileRGBDReceiver::close() {
    return 0;
}

RGBDData *FileRGBDReceiver::getData() {
    RGBDData *data = nullptr;
    int size = 0;
    while (true) {
        unique_lock<mutex> lock(m);
        size = buffer.size();
        if (size > 0) {
            data = buffer.front();
            buffer.pop();
            not_full.notify_one();
            break;
        }
//        cout << "empty" << endl;
    }
    return data;
}

RGBDData *FileRGBDReceiver::_getData() {
    auto t1 = chrono::high_resolution_clock::now();

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

    ThreadPool threadPool(32);

    for (int i = 0; i < 5; ++i) {
        threadPool.spawn(
                [&](int i) {
                    cv::Mat img = cv::imread(
                            path + "video/" + to_string(i + 1) + "-" + to_string(current_idx) + ".png");

                    auto yRange = cv::Range(data->y[i], data->y[i] + data->h_crop[i]);
                    auto xRange = cv::Range(data->x[i], data->x[i] + data->w_crop[i]);
                    memcpy(data->getImage(i), img(yRange, xRange).clone().data, data->w_crop[i] * data->h_crop[i] * 3);
//                    memcpy(data->getImage(i), img.data, 1920 * 1080 * 3);
                }, i);
    }

    for (int i = 0; i < 5; ++i) {
        threadPool.spawn(
                [&](int i) {
                    cv::Mat mask = cv::imread(
                            path + "mask/" + to_string(i + 1) + "-" + to_string(current_idx) + ".png",
                            cv::IMREAD_GRAYSCALE);
                    memcpy(data->getMask(i), mask.data, data->w_crop[0] * data->h_crop[0]);
                }, i);
    }

    for (int i = 0; i < 5; ++i) {
        threadPool.spawn(
                [&](int i) {
                    FILE *f = fopen((path + "depth/" + to_string(i + 1) + "-" + to_string(current_idx) +
                                     ".depth").c_str(),
                                    "rb");
                    fread(data->getDepth(i), data->w_crop[0] * data->h_crop[0] * 4, 1, f);
                    fclose(f);
                }, i);
    }

    threadPool.join();

    auto t2 = chrono::high_resolution_clock::now();
//    cout << current_idx << " "
//         << chrono::duration<double, milli>(t2 - t1).count() / 1000 << " "
//         << 1 / (chrono::duration<double, milli>(t2 - t1).count() / 1000)
//         << endl;

    current_idx = (current_idx + 1) % 90 + 1;
    return data;
}

int FileRGBDReceiver::getBufferSize() {
    int size = 0;
    m.lock();
    size = buffer.size();
    m.unlock();
    return size;
}

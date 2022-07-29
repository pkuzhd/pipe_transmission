#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/RGBDReceiver.h"

using namespace std;

void save(string path, int idx, RGBDData *data) {
    for (int i = 0; i < data->n; ++i) {
        cv::Mat img(data->h[i], data->w[i], CV_8UC3);
        memcpy(img.data, data->getImage(i), data->w[i] * data->h[i] * 3);
        cv::imwrite(path + to_string(idx + 1) + "-" + to_string(i + 1) + ".jpg", img);
    }
}

int main() {
    string test_path = "./test_dir/";
//    for (int j = 0; j < 5; ++j) {
//        for (int i = 0; i < 5; ++i) {
//            cv::Mat img = cv::imread(string("../test_data/" + to_string((j + i) % 5 + 1) + ".jpg"));
//            cv::Mat img_gt(2160, 3840, CV_8UC3);
//            memcpy(img_gt.data, img.data, 2160 * 3840 * 3);
//            cv::imwrite(test_path + to_string(j + 1) + "-" + to_string(i + 1) + "-gt.jpg", img_gt);
//        }
//    }

    RGBDReceiver receiver;
    receiver.open("./pipe_dir/pipe2");

    for (int i = 0; i < 400; ++i) {
        cout << "request data " << i << endl;
        RGBDData *data = receiver.getData();
        while (!data) {
            data = receiver.getData();
        }
        cout << "get data " << i << endl;
        //save(test_path, i, data);
//        delete[] data->imgs;
//        delete[] data->depths;
//        delete[] data->masks;
        delete data;
    }
    receiver.close();
    return 0;
}

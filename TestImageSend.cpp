#include <iostream>
#include "utils/ImageSender.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/time.h>

using namespace std;

void save(string path, int idx, ImageData *data) {
    for (int i = 0; i < data->n; ++i) {
        cv::Mat img(data->h[i], data->w[i], CV_8UC3);
        memcpy(img.data, data->imgs + 2160 * 3840 * 3 * i, data->w[i] * data->h[i] * 3);
        cv::imwrite(path + to_string(idx + 1) + "-" + to_string(i + 1) + ".gt.jpg", img);
    }
}

int main() {
    string test_path = "../test_dir/";

    ImageSender sender;
    sender.open("../pipe_dir/pipe1");
    long time_total = 0;
    long long int bytes_sent = 0;

    for (int j = 0; j < 1000; ++j) {
        std::cout << j << std::endl;
        ImageData *data = new ImageData;
        data->n = 5;
        data->w = new int[data->n];
        data->h = new int[data->n];
        data->imgs = new char[data->n * 2160 * 3840 * 3];
        for (int i = 0; i < 5; ++i) {
            data->h[i] = 2160;
            data->w[i] = 3840;
            cv::Mat img = cv::imread("../test_data/" + std::to_string(i + 1) + ".jpg");
            memcpy(data->imgs + 2160 * 3840 * 3 * i, img.data, 2160 * 3840 * 3);
        }
        save(test_path, j, data);
        struct timeval tv;
        gettimeofday(&tv, NULL);
        long now1 = tv.tv_sec * 1000 + tv.tv_usec / 1000;
        int len = sender.sendData(data);
        gettimeofday(&tv, NULL);
        long now2 = tv.tv_sec * 1000 + tv.tv_usec / 1000;
        std::cout << "time spent: " << now2 - now1 << std::endl;
        time_total += now2 - now1;
        bytes_sent += len;
        cout << bytes_sent << "B" << " ";
        cout << time_total << "ms" << " ";
        cout << bytes_sent / time_total / 1000 << "MB/s" << endl;

    }

    sender.close();
    return 0;
}

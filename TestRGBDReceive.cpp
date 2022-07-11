#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "receiver/RGBDReceiver.h"

using namespace std;

void save(string path, int idx, RGBDData *data) {

}

int main() {
    string test_path = "";
    RGBDReceiver receiver;
    receiver.open("pipe filename");

    for (int i = 0; i < 5; ++i) {
        cout << "request data " << i << endl;
        RGBDData *data = receiver.getData();
        while (!data) {
            data = receiver.getData();
        }
        cout << "get data " << i << endl;
        save(test_path, i, data);
        delete[] data->imgs;
        delete[] data->depths;
        delete[] data->masks;
        delete data;
    }
    receiver.close();
    return 0;
}

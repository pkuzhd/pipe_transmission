import cv2
import os
import numpy as np

class ImageData():
    def __init__(self, N, imgs):
        self.N = N
        self.imgs = imgs


class ImageReceiver():
    def __init__(self):
        self.rf = 0

    def open(self, filename):
        if not os.path.exists(filename):
            os.mkfifo(filename)
        self.rf = os.open(filename, os.O_RDONLY)
        print("os.open finished")

    def close(self):
        os.close(self.rf)

    def getData(self) -> ImageData:
        image_data = ImageData(0, [])
        buf = os.read(self.rf, 1)
        image_data.N = int.from_bytes(buf, byteorder='little', signed=False)
        print(image_data.N)
        for i in range(image_data.N):
            buf = os.read(self.rf, 4)
            w = int.from_bytes(buf, byteorder='little', signed=False)
            buf = os.read(self.rf, 4)
            h = int.from_bytes(buf, byteorder='little', signed=False)
            print(w, h)
            buf = os.read(self.rf, w * h * 3)
            image_data.imgs.append(np.frombuffer(buf, np.uint8).reshape((h, w, 3)))

        return image_data





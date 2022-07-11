import cv2


class ImageData():
    def __init__(self, N, imgs):
        self.N = N
        self.imgs = imgs


class ImageReceiver():
    def __init__(self):
        pass

    def open(self, filename):
        pass

    def close(self):
        pass

    def getData(self) -> ImageData:
        pass

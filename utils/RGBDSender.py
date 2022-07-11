import cv2


class RGBDData():
    def __init__(self, N, imgs, depths, masks, crops):
        self.N = N
        self.imgs = imgs
        self.depths = depths
        self.masks = masks
        self.crops = crops


class RGBDSender():
    def __init__(self):
        pass

    def open(self, filename):
        pass

    def close(self):
        pass

    def sendData(self, data: RGBDData):
        pass

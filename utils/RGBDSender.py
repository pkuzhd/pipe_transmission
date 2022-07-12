import os
import json

class RGBDData():
    def __init__(self, N, imgs, depths, masks, crops):
        self.N = N
        self.imgs = imgs
        self.depths = depths
        self.masks = masks
        self.crops = crops


class RGBDSender():
    def __init__(self):
        self.wf = 0

    def open(self, filename):
        self.wf = os.open(filename, os.O_SYNC | os.O_CREAT | os.O_RDWR)


    def close(self):
        os.close(self.wf)

    def sendData(self, data: RGBDData):
        msg = json.dumps(data.__dict__)
        len_send = os.write(self.wf, msg)

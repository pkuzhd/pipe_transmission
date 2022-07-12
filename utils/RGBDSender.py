import os
import numpy as np
import json

class RGBDData():
    def __init__(self, N, imgs, depths, masks, crops):
        self.N = N
        self.crops = crops  # 5 * 4 * 4
        self.imgs = imgs  # N * W * H * 3
        self.depths = depths # N * W_crop * H_crop * 4
        self.masks = masks # N * W_crop * H_crop * 1



class RGBDSender():
    def __init__(self):
        self.wf = 0

    def open(self, filename):
        self.wf = os.open(filename, os.O_SYNC | os.O_CREAT | os.O_RDWR)


    def close(self):
        os.close(self.wf)

    def sendData(self, data: RGBDData):
        #send N
        msg = data.N.to_bytes(1, "little")
        for i in range(data.N):
            # send crops
            for j in range(4):
                msg += data.crops[i][j].to_bytes(4, "little")
        print(msg)

        for i in range(data.N):
            # send imgs
            w = data.crops[i][0]
            h = data.crops[i][1]
            msg += data.imgs[i].data.tobytes()

        for i in range(data.N):
            # send depths
            w_crop = data.crops[i][0]
            h_crop = data.crops[i][1]
            msg += data.depths[i].data.tobytes()

        for i in range(data.N):
            # send masks
            w_crop = data.crops[i][0]
            h_crop = data.crops[i][1]
            msg += data.masks[i].data.tobytes()
        len_send = os.write(self.wf, msg)
        print(f"length of msg is : {len_send}")



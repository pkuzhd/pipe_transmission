import os
import numpy as np
import json
import datetime

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
        self.wf = os.open(filename, os.O_WRONLY)


    def close(self):
        os.close(self.wf)

    def sendData(self, data: RGBDData):



        #send N
        msg = data.N.to_bytes(1, "little")

        # send crops
        for i in range(data.N):
            for j in range(4):
                msg += data.crops[i][j].to_bytes(4, "little")
        #print(msg)

        len_send = os.write(self.wf, msg)

        # send imgs
        for i in range(data.N):
            w = data.crops[i][0]
            h = data.crops[i][1]
            len_send += os.write(self.wf, data.imgs[i].data.tobytes())

        # send depths
        for i in range(data.N):
            w_crop = data.crops[i][0]
            h_crop = data.crops[i][1]
            len_send += os.write(self.wf, data.depths[i].data.tobytes())

        # send masks
        for i in range(data.N):
            w_crop = data.crops[i][0]
            h_crop = data.crops[i][1]
            len_send += os.write(self.wf, data.masks[i].data.tobytes())


        print(f"length of msg is : {len_send}")
        return len_send




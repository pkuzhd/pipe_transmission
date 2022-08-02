from utils.MultiProcessBuffer import MultiProcessBuffer
from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
import utils.RecvDataProcess
import utils.SendDataProcess
import numpy as np
import cv2
import os
from utils.ImageReceiver import ImageData
from utils.RGBDSender import RGBDData
import time


class DepthProcess():
    SendDataProcess: utils.SendDataProcess.SendDataProcess
    depthEstimations: DepthEstimation_forRGBD
    RecvDataProcess: utils.RecvDataProcess.RecvDataProcess

    def __init__(self, RecvDataProcess, SendDataProcess, depthEstimations):
        self.RecvDataProcess = RecvDataProcess
        self.SendDataProcess = SendDataProcess
        self.depthEstimations = depthEstimations

    def run(self, test_num):
        j = 0
        print(f"深度进程{os.getpid()}")
        t = [0, 0, 0, 0]
        times = 0
        while j < test_num:
            j += 1
            time1 = time.time()
            listRGB = self.RecvDataProcess.runRGB()
            t2 = time.time()
            # self.SendDataProcess.downloadRGB(j)
            rgbd_data = self.depthEstimations.getRGBD(ImageData(5, listRGB), crop=True)
            rgbd_data = RGBDData(
                rgbd_data["num_view"],
                rgbd_data["imgs"],
                rgbd_data["depths"],
                rgbd_data["masks"],
                rgbd_data["crops"]
            )
            # rgbd_data = RGBDData(
            #     5,
            #     np.zeros((5, 1080, 1920, 3), dtype=np.uint8),
            #     np.zeros((5, 896, 768), dtype=np.float32),
            #     np.zeros((5, 896, 768, 1), dtype=np.uint8),
            #     [(896, 768, 0, 0), (896, 768, 0, 0), (896, 768, 0, 0), (896, 768, 0, 0), (896, 768, 0, 0)],
            # )
            t3 = time.time()
            self.SendDataProcess.runView(np.array([rgbd_data.N], dtype=np.uint8))
            self.SendDataProcess.runRGBtoPipe(rgbd_data.imgs)
            self.SendDataProcess.runDepth(np.expand_dims(rgbd_data.depths,axis=3))
            self.SendDataProcess.runMask(rgbd_data.masks)
            self.SendDataProcess.runCrop(rgbd_data.crops)
            t4 = time.time()
            times = times + t3 - t2
            # print("R->D = ",t2 - time1," Depth = ",t3 - t2,"D->B = ",t4 - t3)
            t[0] += t2 - time1
            t[1] += t3 - t2
            t[2] += t4 - t3
            t[3] += t4 - time1
            # print(j-1,"Depth = ",t3 - t2)
            print(j, t2 - time1, t3 - t2, t4 - t3, t4 - time1)
        print(t[0] / test_num, t[1] / test_num, t[2] / test_num, t[3] / test_num)
        print(test_num / t[0], test_num / t[1], test_num / t[2], test_num / t[3])
        # print("average depth = ",times/test_num)

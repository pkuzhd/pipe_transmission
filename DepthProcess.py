from utils.MultiProcessBuffer import MultiProcessBuffer
from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
import utils.RecvDataProcess
import utils.SendDataProcess
import numpy as np
import cv2
import os
from utils.ImageReceiver import ImageData
from utils.RGBDSender import RGBDData
class DepthProcess():
    
    SendDataProcess: utils.SendDataProcess.SendDataProcess
    depthEstimations: DepthEstimation_forRGBD
    RecvDataProcess:utils.RecvDataProcess.RecvDataProcess
    
    def __init__(self,RecvDataProcess,SendDataProcess,depthEstimations):
        self.RecvDataProcess = RecvDataProcess
        self.SendDataProcess = SendDataProcess
        self.depthEstimations = depthEstimations
    
    def run(self):
        j = 0
        print(f"深度进程{os.getpid()}")
        while True:
            listRGB = self.RecvDataProcess.runRGB()
            #self.SendDataProcess.downloadRGB(j)
            rgbd_data = self.depthEstimations.getRGBD(ImageData(5, listRGB),crop = True)
            rgbd_data = RGBDData(
                rgbd_data["num_view"],
                rgbd_data["imgs"],
                rgbd_data["depths"],
                rgbd_data["masks"],
                rgbd_data["crops"]
            )
            
            self.SendDataProcess.runView(np.array([rgbd_data.N],dtype=np.uint8))   
            self.SendDataProcess.runRGBtoPipe(rgbd_data.imgs)
            self.SendDataProcess.runDepth(rgbd_data.depths)
            self.SendDataProcess.runMask(rgbd_data.masks)
            self.SendDataProcess.runCrop(rgbd_data.crops)
            
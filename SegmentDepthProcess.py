import imp
import time
from utils.SegmentBufferModule import SegmentBufferModule
from utils.SegmentReceiver import SegmentReceiver
from utils.SegmentSender import SegmentSender
from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
from utils.ImageReceiver import ImageData
from utils.RGBDSender import RGBDData
import numpy as np
class SegmentDepthProcess:
    
    
    def  __init__(self,segmentSender,segmentReceiver,depthEstimation):
        self.sender : SegmentSender
        self.receiver : SegmentReceiver
        
        self.sender = segmentSender
        self.receiver = segmentReceiver
        self.depthEstimation = depthEstimation
        self.enumImg = ["imgs","depths","masks"]
        
    def depthProcess(self,test_num):
        times = [0,0,0]
        testCount = 0
        while testCount < test_num:
            testCount += 1
            # bytes N whcd crops 
            # imgsData
            t1 = time.time()
            RGBimgs,imgNum = self.sender.getImagefromBufferA()
            t2 = time.time()
            rgbd_data = self.depthEstimation.getRGBD(ImageData(imgNum, RGBimgs), crop=True)
            t3 = time.time()   
            rgbd_data["depths"] = np.expand_dims(rgbd_data["depths"],axis=3)
            metaData = np.array([4,imgNum],dtype=np.uint32)
            imgs = []
            for i in self.enumImg:
                if i == "depths":
                    img,metaData = SegmentReceiver.encodingDepth(rgbd_data[i],metaData)
                    imgs.append(img)
                else:
                    img,metaData = SegmentReceiver.encodingImage(rgbd_data[i],metaData)
                    imgs.append(img)
            metaData[0] += 4 * len(rgbd_data["crops"]) * 4
            for i in rgbd_data["crops"]:
                metaData = np.concatenate([metaData,i],axis=0)
            self.receiver.bufferManager.recv(self.receiver.nameB,metaData.astype(np.uint32),metaData.shape[0])
            for inputImage in imgs:
                self.receiver.bufferManager.recv(self.receiver.nameB,inputImage,inputImage.shape[0])
            t4 = time.time()
            print(testCount,t2-t1,t3-t2,t4-t3)
            times[0] = times[0] + t2 - t1
            times[1] = times[1] + t3 - t2
            times[2] = times[2] + t4 - t3
        print("1 ",times[0]/test_num,"2 ",times[1]/test_num,"3 ",times[2]/test_num)
        self.receiver.bufferManager.unlinks()    
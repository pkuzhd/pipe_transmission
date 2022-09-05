from SegmentBufferManager import SegmentBufferManager
from ImageReceiver import ImageData, ImageReceiver
import numpy as np
import cv2
from SegmentBufferModule import SegmentBufferModule
import time
from RGBDSender import RGBDSender,RGBDData
class SegmentSender:
    bufferManager : SegmentBufferManager
    
    def __init__(self,SegmentBufferManager,SenderName):
        self.bufferManager = SegmentBufferManager
        # self.pipeReceiver = ImageReceiver()
        # self.pipeReceiver.open(Receivername)
        self.nameA = self.bufferManager.getNameA()
        self.nameB = self.bufferManager.getNameB()
        self.pipeSender = RGBDSender()
        self.SenderName = SenderName
        
    def getImagefromBufferA(self):
        byteNum = self.bufferManager.send(self.nameA,1,np.uint32)
        Bufferoffset = 0
        outputData = self.bufferManager.send(self.nameA,byteNum[0],np.uint8)
        imgs = []
        imgNum = np.ndarray(1,dtype=np.uint32,buffer=outputData.data,offset=Bufferoffset)
        Bufferoffset += 4
        imgShape = np.ndarray(imgNum[0] * 4,dtype=np.uint32,buffer=outputData.data,offset=Bufferoffset)
        Bufferoffset += 16 * imgNum[0] 
        for i in range(imgNum[0]):
            img = np.ndarray((imgShape[i * 4 + 0],imgShape[i * 4 + 1],imgShape[i * 4 +  2]),dtype=SegmentBufferModule.getdtype(imgShape[i * 4 + 3]),buffer=outputData.data,offset=Bufferoffset)
            Bufferoffset += img.nbytes
            imgs.append(img)
        return imgs,imgNum[0]
    
    def getDepthfromBufferB(self,test_num):
        testCount = 0
        self.pipeSender.open(filename=self.SenderName)
        while testCount < test_num:
            testCount += 1 
            byteNum = self.bufferManager.send(self.nameB,1,np.uint32)
            Bufferoffset = 0
            outputData = self.bufferManager.send(self.nameB,byteNum[0],np.uint8)
            imgs = []
            Depths = []
            Masks = []
            crops = []
            imgNum = np.ndarray(1,dtype=np.uint32,buffer=outputData.data,offset=Bufferoffset)
            Bufferoffset += 4
            imgShape = np.ndarray(imgNum[0] * 4 * 3,dtype=np.uint32,buffer=outputData.data,offset=Bufferoffset)
            Bufferoffset += 16 * imgNum[0] * 3 
            for i in range(imgNum[0]):
                crops.append(np.ndarray([4],dtype=np.uint32,buffer=outputData.data,offset=Bufferoffset))
                Bufferoffset += 16
                
            for i in range(imgNum[0]):
                img = np.ndarray((imgShape[i * 4 + 0],imgShape[i * 4 + 1],imgShape[i * 4 +  2]),dtype=SegmentBufferModule.getdtype(imgShape[i * 4 + 3]),buffer=outputData.data,offset=Bufferoffset)
                Bufferoffset += img.nbytes
                imgs.append(img)
                
            for i in range(imgNum[0],imgNum[0] * 2):
                img = np.ndarray((imgShape[i * 4 + 0],imgShape[i * 4 + 1],imgShape[i * 4 +  2]),dtype=SegmentBufferModule.getdtype(imgShape[i * 4 + 3]),buffer=outputData.data,offset=Bufferoffset)
                Bufferoffset += img.nbytes
                Depths.append(img)
                
            for i in range(imgNum[0]* 2 , imgNum[0] * 3):
                img = np.ndarray((imgShape[i * 4 + 0],imgShape[i * 4 + 1],imgShape[i * 4 +  2]),dtype=SegmentBufferModule.getdtype(imgShape[i * 4 + 3]),buffer=outputData.data,offset=Bufferoffset)
                Bufferoffset += img.nbytes
                Masks.append(img)
                
            senddatas = RGBDData(int(imgNum[0]),imgs,Depths,Masks,crops)
            
            # for i in range(imgNum[0]):
            #     cv2.imwrite("/home/pku/view_synthesis/software/multiProcess/pipe_transmission/utils/testImage/img"+str(testCount)+"-"+str(i)+".png",imgs[i])
            #     Depths[i] = (Depths[i] - np.min(Depths[i])) / (np.max(Depths[i]) - np.min(Depths[i])) * 255
            #     cv2.imwrite("/home/pku/view_synthesis/software/multiProcess/pipe_transmission/utils/testImage/Depth"+str(testCount)+"-"+str(i)+".png",Depths[i])
            #     cv2.imwrite("/home/pku/view_synthesis/software/multiProcess/pipe_transmission/utils/testImage/Mask"+str(testCount)+"-"+str(i)+".png",Masks[i])
           
           
           #print("epoch ",testCount,"views = ",imgNum[0])   

            self.pipeSender.sendData(senddatas)
        
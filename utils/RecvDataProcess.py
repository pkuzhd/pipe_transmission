from MultiProcessBuffer import MultiProcessBuffer
import cv2
import os
import numpy as np
import time
from RGBDSender import RGBDData,RGBDSender
class RecvDataProcess():
    
    MPB: MultiProcessBuffer
    
    def __init__(self, MPB , filenames):
        self.MPB = MPB
        self.pipeSender = RGBDSender()
        self.pipeSender.open(filename=filenames)

    def runRGB(self):
        listRGB = []
        t0 = 0
        t4 = 0
        for i in range(5):
            t1 = time.time()
            t2 = time.time()
            listRGB.append(self.MPB.readRGB().copy())
            t3 = time.time()
            t0 = t0 + t2 - t1
            t4 = t4 + t3 - t2
        print("copy = ",t4/5,"read = ",t0/5)
        return listRGB
    
    def runRGBtoPipe(self):
        listRGB = []
        for i in range(5):
            RGBs = self.MPB.readRGBtoPipe().copy()
            listRGB.append(RGBs)
        return listRGB
    
    def runDepth(self):
        listDepth = []
        for i in range(5):
            Depth = self.MPB.readDepth().copy()
            listDepth.append(Depth)
        return listDepth
            
    def runMask(self):
        listMask = []
        for i in range(5):
            Mask = self.MPB.readMask().copy()
            listMask.append(Mask)
        return listMask

    def runCrop(self):
        listCrop = []
        for i in range(5):
            Crop = self.MPB.readCrop().copy()
            listCrop.append(Crop[:])
        return listCrop

    def runView(self):
        return self.MPB.readView().copy()
    
    def runSender(self,test_num):
        docu = "/home/pku/view_synthesis/software/multiProcess/datas/test/"
        k = 0
        times = 0
        while k < test_num:
            k+=1
            t1 = time.time()
            view = self.runView()
            imgs = self.runRGBtoPipe()
            depths = self.runDepth()
            masks = self.runMask()
            crops= self.runCrop()
            # for i in range(5):
            #     cv2.imwrite(docu+"imgs"+str(k)+str(i)+".png",imgs[i])
            #     depths[i] = ( depths[i] - np.min( depths[i])) / (np.max( depths[i]) - np.min( depths[i])) * 255
            #     cv2.imwrite(docu+"depth"+str(k)+str(i)+".png",depths[i])
            #     cv2.imwrite(docu+"mask"+str(k)+str(i)+".png",masks[i])
            # print("view = ",view[0])
            # print("imgs = ",imgs[0])
            # print("depths =",depths[0])
            # print("masks =",masks[0])
            # print("crops = ",crops[0])
            senddatas = RGBDData(int(view[0]),imgs,depths,masks,crops)
            t2 = time.time()
            self.pipeSender.sendData(senddatas)
            t3 = time.time()
            times = times + t2 - t1
            # print(k-1,"B -> P : ",t2 - t1)
        # print("average all = ",times/test_num)
        self.MPB.freeRGBBuffer()
        self.MPB.freeDepthBuffer()
        self.MPB.freeMaskBuffer()
        self.MPB.freeCropBuffer()
        self.MPB.freeViewBuffer()
        self.MPB.freeRGBtoPipeBuffer()
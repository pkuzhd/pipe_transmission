from MultiProcessBuffer import MultiProcessBuffer
import cv2
import os
import numpy as np
class RecvDataProcess():
    
    MPB: MultiProcessBuffer
    
    def __init__(self, MPB):
         self.MPB = MPB
         savePath = "datas/"
         
    def runRGB(self):
        listRGB = []
        for i in range(5):
            RGBs = self.MPB.readRGB().copy()
            print(os.getpid(),"readout image ",str(i))
            listRGB.append(RGBs)
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
    
    def runSender(self):
        docu = "/home/pku/view_synthesis/software/multiProcess/datas/test/"
        k = 0
        while k < 10:
            k+=1
            view = self.runView()
            imgs = self.runRGBtoPipe()
            depths = self.runDepth()
            masks = self.runMask()
            crops = self.runCrop()
            for i in range(5):
                cv2.imwrite(docu+"imgs"+str(k)+str(i)+".png",imgs[i])
                depths[i] = ( depths[i] - np.min( depths[i])) / (np.max( depths[i]) - np.min( depths[i])) * 255
                cv2.imwrite(docu+"depth"+str(k)+str(i)+".png",depths[i])
                cv2.imwrite(docu+"mask"+str(k)+str(i)+".png",masks[i]) 
                
            senddatas = [view,imgs,depths,masks,crops]
            print("write to pipes")
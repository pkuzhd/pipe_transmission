from MultiProcessBuffer import MultiProcessBuffer
from multiprocessing import Process
from ImageReceiver import ImageData, ImageReceiver
import cv2
import os
import time
class SendDataProcess():

    MPB : MultiProcessBuffer
    
    def __init__(self, MPB,pipePath):
        self.MPB = MPB
        self.pipePath = pipePath
        self.pipeReceiver = ImageReceiver()
        self.pipeReceiver.open(pipePath)

    def runRGB(self,num_test):
        j = 0
        times = 0
        print(f"读input进程{os.getpid()} is running.")
        while j < num_test:
            j += 1
            t1 = time.time()
            imageRGB = self.pipeReceiver.getData().imgs.copy()
            t2 =time.time()
          #  imageRGB = [cv2.imread(self.pipePath + "imgs/" + str(i) + "-1.png") for i in range(1,6)]
            for i in range(5):
                self.MPB.writeRGB(imageRGB[i].copy())
            t3 = time.time()
            times = times + t3 - t1
            # print(j,"P->M = ",t2-t1,"M -> B = ",t3-t2)
        # print("average p->b = ",times/num_test)
        self.pipeReceiver.close()
            #print(j,"P -> B = ",time.time() - t1 )

    def runRGBtoPipe(self,imageRGBtoPipe):
        for i in range(5):
                self.MPB.writeRGBtoPipe(imageRGBtoPipe[i])
                
    def runDepth(self,imageDepth):
            for i in range(5):
                self.MPB.writeDepth(imageDepth[i])

    def runMask(self,imageMask):
            for i in range(5):
                self.MPB.writeMask(imageMask[i])
    
    def runCrop(self,imageCrop):
        for i in range(5):
                self.MPB.writeCrop(imageCrop[i])
    
    def runView(self,numView):
        self.MPB.writeView(numView)
from MultiProcessBuffer import MultiProcessBuffer
from multiprocessing import Process
from ImageReceiver import ImageData, ImageReceiver
import cv2
import os

class SendDataProcess():

    MPB : MultiProcessBuffer
    
    def __init__(self, MPB,pipePath):
        self.MPB = MPB
        self.pipePath = pipePath
        # self.pipeReceiver = ImageReceiver()
        # self.pipeReceiver.open(pipePath)

    def runRGB(self):
        j = 0
        print(f"读input进程{os.getpid()} is running.")
        while True:
            imageRGB = [cv2.imread(self.pipePath + "imgs/" + str(i) + "-1.png") for i in range(1,6)]   
            for i in range(5):
                self.MPB.writeRGB(imageRGB[i].copy())
                print("进程",os.getpid(),"input image ",i)
            
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
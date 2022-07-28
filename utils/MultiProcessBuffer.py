
from bufferModule import BufferModule
import numpy as np
import multiprocessing
import os
import time
class MultiProcessBuffer:

    def __init__(self, nameRGB,nameDepth,nameMask,widthRGB,heightRGB,channelsRGB,RGBdtypes,RGBnbytes,widthDepth,heightDepth,channelsDepth,\
                Depthdtypes,Depthnbytes,widthMask,heightMask,channelsMask,Maskdtypes,Masknbytes,\
                nameCrop,widthCrop,heightCrop,channelsCrop,Cropdtype,Cropnbytes,\
                nameView,widthView,heightView,channelsView,Viewdtype,Viewnbytes,bufferSize):
        
        self.bufferRGB = BufferModule(nameRGB,widthRGB,heightRGB,channelsRGB,RGBdtypes,RGBnbytes,bufferSize)
        self.bufferDepth = BufferModule(nameDepth,widthDepth,heightDepth,channelsDepth,Depthdtypes,Depthnbytes,bufferSize)
        self.bufferMask = BufferModule(nameMask,widthMask,heightMask,channelsMask,Maskdtypes,Masknbytes,bufferSize)
        self.bufferCrop = BufferModule(nameCrop,widthCrop,heightCrop,channelsCrop,Cropdtype,Cropnbytes,bufferSize)
        self.bufferView = BufferModule(nameView,widthView,heightView,channelsView,Viewdtype,Viewnbytes,1)
        self.bufferRGBtoPipe = BufferModule(nameRGB+"pipe",widthRGB,heightRGB,channelsRGB,RGBdtypes,RGBnbytes,bufferSize)
        
        self.outputRGB = np.zeros((heightRGB,widthRGB,channelsRGB),dtype = RGBdtypes)
        self.outputDepth = np.zeros((heightDepth,widthDepth,channelsDepth),dtype = Depthdtypes)
        self.outputMask = np.zeros((heightMask,widthMask,channelsMask),dtype = Maskdtypes)
        self.outputCrop = np.zeros((channelsCrop),dtype = Cropdtype)
        self.outputView = np.zeros(channelsView,dtype=Viewdtype)
        self.outputRGBtoPipe = np.zeros((heightRGB,widthRGB,channelsRGB),dtype = RGBdtypes)
    
    
    def readRGBtoPipe(self):
        return self.bufferRGBtoPipe.readData(self.outputRGBtoPipe)

    def writeRGBtoPipe(self,inputImage):
        return self.bufferRGBtoPipe.writeData(inputImage)
    
    def readRGB(self):
        return self.bufferRGB.readData(self.outputRGB)

    def writeRGB(self,inputImage):
        return self.bufferRGB.writeData(inputImage)

    def downloadRGB(self,path,j):
        self.bufferRGB.imwrite(path,j)
    
    def readDepth(self):
        return self.bufferDepth.readData(self.outputDepth)

    def writeDepth(self,inputImage):
        return self.bufferDepth.writeData(inputImage)
    
    def downloadDepth(self,path,j):
        self.bufferDepth.imwrite(path,j)

    def readMask(self):
        return self.bufferMask.readData(self.outputMask)

    def writeMask(self,inputImage):
        return self.bufferMask.writeData(inputImage)
    
    def readCrop(self):
        return self.bufferCrop.readData(self.outputCrop)
    
    def writeCrop(self,inputImage):
        return self.bufferCrop.writeData(inputImage)
    
    def readView(self):
        return self.bufferView.readData(self.outputView)
    
    def writeView(self,inputImage):
        return self.bufferView.writeData(inputImage)

    def freeRGBBuffer(self):
        self.bufferRGB.unlinkShm()

    def freeDepthBuffer(self):
        self.bufferDepth.unlinkShm()

    def freeMaskBuffer(self):
        self.bufferMask.unlinkShm()
        
    def freeCropBuffer(self):
        self.bufferCrop.unlinkShm()
        
    def freeViewBuffer(self):
        self.bufferView.unlinkShm()

def sendRGB(buffers,inputImage):
    print(f"sender进程{os.getpid()} begin")
    iters = 0
    while iters < 1000:
        time1 = time.time()
        buffers.writeRGB(inputImage)
        #print(f"RGBwriter进程{os.getpid()}从queue中获取到iters = {iters}, dataTime = {time.time() - time1},带宽v = { (1920*1080*3/1024/1024) / (time.time() - time1)} MBps")
        iters += 1   

def recvRGB(buffers):
    print(f"reader进程{os.getpid()} begin")
    times = 0
    iters = 0
    while iters < 1000:
        time1 = time.time()
        buffers.outputRGB = buffers.readRGB()  
        iters += 1 
        times += time.time() - time1 
    print(f"RGBreader进程{os.getpid()}从queue中获取到iters = {iters}, dataTime = {times/1000},带宽v = { (1920*1080*3/1024/1024) / (times/1000)} MBps")   

def sendDepth(buffers,inputImage):
    print(f"sender进程{os.getpid()} begin")
    iters = 0
    while iters < 1000:
        time1 = time.time()
        buffers.writeDepth(inputImage)
        #print(f"Depthwriter进程{os.getpid()}从queue中获取到iters = {iters}, dataTime = {time.time() - time1},带宽v = { (1920*1080*1*4/1024/1024) / (time.time() - time1)} MBps")
        iters += 1   
    
def recvDepth(buffers):
    print(f"reader进程{os.getpid()} begin")
    times = 0
    iters = 0
    while iters < 1000:
        time1 = time.time()
        buffers.outputDepth = buffers.readDepth()    
        iters += 1 
        times += time.time() - time1
    print(f"Depthreader进程{os.getpid()}从queue中获取到iters = {iters}, dataTime = {times/1000},带宽v = { (1920*1080*1*4/1024/1024) / (times/1000)} MBps") 

def sendMask(buffers,inputImage):
    print(f"sender进程{os.getpid()} begin")
    iters = 0
    while iters < 1000:
        time1 = time.time()
        buffers.writeDepth(inputImage)
        #print(f"Maskwriter进程{os.getpid()}从queue中获取到iters = {iters}, dataTime = {time.time() - time1},带宽v = { (1920*1080*1*8/1024/1024) / (time.time() - time1)} MBps")
        iters += 1   

def recvMask(buffers):
    print(f"reader进程{os.getpid()} begin")
    iters = 0
    times = 0
    while iters < 1000:
        time1 = time.time()
        buffers.outputDepth = buffers.readDepth()  
        iters += 1 
        times += time.time() - time1
    print(f"Maskreader进程{os.getpid()}从queue中获取到iters = {iters}, dataTime = {times/1000},带宽v = { (1920*1080*1*8/1024/1024) / (times/1000)} MBps")   




if __name__ == "__main__":
    print(f"main process {os.getpid()} begin..")

    nameRGB = "bufferRGB"
    nameDepth = "bufferDepth"
    nameMask = "bufferMask"

    bufferSize = 5

    widthRGB = 1920
    heightRGB = 1080
    channelsRGB = 3
    RGBdtypes = np.uint8
    inputRGB = np.zeros((heightRGB,widthRGB,channelsRGB),dtype=RGBdtypes)
    RGBnbytes = inputRGB.nbytes

    widthDepth = 1920
    heightDepth = 1080
    channelsDepth = 1
    Depthdtypes = np.float32
    inputDepth = np.zeros((heightDepth,widthDepth,channelsDepth),dtype=Depthdtypes)
    Depthnbytes = inputDepth.nbytes

    widthMask = 1920
    heightMask = 1080
    channelsMask = 1
    Maskdtypes = np.float64
    inputMask = np.zeros((heightMask,widthMask,channelsMask),dtype=Maskdtypes)
    Masknbytes = inputMask.nbytes

    MPB = MultiProcessBuffer(nameRGB,nameDepth,nameMask,widthRGB,heightRGB,channelsRGB,RGBdtypes,RGBnbytes,widthDepth,heightDepth,channelsDepth,\
                Depthdtypes,Depthnbytes,widthMask,heightMask,channelsMask,Maskdtypes,Masknbytes,bufferSize)
    
    psendRGB = multiprocessing.Process(target = sendRGB,args = (MPB,inputRGB))
    psendDepth = multiprocessing.Process(target = sendDepth,args = (MPB,inputDepth))
    psendMask = multiprocessing.Process(target = sendMask,args = (MPB,inputMask))
    
    preadRGB = multiprocessing.Process(target = recvRGB,args = (MPB,))
    preadDepth = multiprocessing.Process(target = recvDepth,args = (MPB,))
    preadMask = multiprocessing.Process(target = recvMask,args = (MPB,))

    psendRGB.start()
    preadRGB.start()
    psendDepth.start()
    preadDepth.start()
    psendMask.start()
    preadMask.start()

    psendRGB.join()
    preadRGB.join()
    psendDepth.join()
    preadDepth.join()
    preadMask.join()
    psendMask.join()

    MPB.freeRGBBuffer()
    MPB.freeDepthBuffer()
    MPB.freeMaskBuffer()

    print(f"main process {os.getpid()} end..")
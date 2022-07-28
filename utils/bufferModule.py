from ctypes import sizeof
from multiprocessing import Process,Lock,shared_memory,Semaphore
import os,time
from venv import create
import numpy as np
import cv2

class BufferModule:
    def __init__(self,names,width,height,channels,dtypes,nbytes,bufferSize):
        #memory ID
        try:
            self.shm = shared_memory.SharedMemory(name = names, create=True,size = (bufferSize + 1) * nbytes * BufferModule.getshmBuffer(dtypes) + 2 )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name = names, create=False,size = (bufferSize + 1) * nbytes * BufferModule.getshmBuffer(dtypes) + 2 )
        
        #buffer
        if (height * width * channels != 0):
            self.memorySmallBuffer = np.ndarray(((bufferSize+1) * height * width * channels * BufferModule.getshmBuffer(dtypes) + 2  ) ,  dtype=np.uint8, buffer=self.shm.buf)
            self.memoryBuffer = np.ndarray((bufferSize+1,height,width,channels),dtype = dtypes,buffer = self.memorySmallBuffer.data,offset = 2)
        
        else: # views
            self.memorySmallBuffer = np.ndarray(((bufferSize+1) * BufferModule.getshmBuffer(dtypes) * nbytes + 2  ) ,  dtype=np.uint8, buffer=self.shm.buf)
            self.memoryBuffer = np.ndarray((bufferSize+1,channels),dtype = dtypes,buffer = self.memorySmallBuffer.data,offset = 2)
        
        #读者写者问题的三个锁
        self.lockMutex = Lock()
        self.lockReader = Semaphore(0)
        self.lockWriter = Semaphore(bufferSize)
        #queue top
        self.front = np.ndarray((1) , dtype = np.uint8 ,buffer=self.memorySmallBuffer.data,offset=1)
        self.rear = np.ndarray((1) , dtype = np.uint8 , buffer=self.memorySmallBuffer.data,offset=0)
        self.front[0] = 0
        self.rear[0] = 0
        self.bufferSize = bufferSize
    
            
    def writeData(self,inputImage):
            self.lockWriter.acquire()
            self.lockMutex.acquire()
            self.memoryBuffer[self.rear[0],:] = inputImage[:]
            self.rear[0] = (self.rear[0] + 1) % (self.bufferSize+1)
            self.lockMutex.release()
            self.lockReader.release()
    
    def readData(self,outputImage):
            self.lockReader.acquire()
            self.lockMutex.acquire()
            #print("[read] front = ",self.front)
            outputImage[:] = self.memoryBuffer[self.front[0],:]
            self.front[0] = (self.front[0] + 1 ) % (self.bufferSize+1)
            self.lockMutex.release()
            self.lockWriter.release()
            return outputImage

    def unlinkShm(self):
        self.shm.unlink()

    def isEmpty(self):
        return (self.front[0] == self.rear[0])

    def isFull(self):
        return (self.front[0] == (self.rear[0] + 1) % (self.bufferSize+1))
    
    def imwrite(self,path,j):
        fronts = self.front[0]
        while ((fronts == self.rear[0]) == False):
            # cv2.imshow("output"+str(j)+str(fronts),self.memoryBuffer[fronts])
            # cv2.waitKey()
            cv2.imwrite(path+str(fronts)+".png",self.memoryBuffer[fronts])
            fronts = (fronts + 1) % (self.bufferSize+1) 
    
    def getshmBuffer(dtype):
        if(dtype == np.uint8):
            return 1
        elif (dtype == np.float32):
            return 4
        elif (dtype == np.float64):
            return 8
        elif (dtype == np.uint32):
            return 4
        else:
            raise NotImplementedError
        
# def send(buffers,inputImage):
#     print(f"sender进程{os.getpid()} begin")
#     iters = 0
#     while iters < 1000:
#         time1 = time.time()
#         buffers.writeData(inputImage)
#         print(f"writer进程{os.getpid()}从queue中获取到iters = {iters}, dataTime = {time.time() - time1},带宽v = { (1024) / (time.time() - time1)} MBps")
#         iters += 1   

# def recv(buffers):
#    print(f"reader进程{os.getpid()} begin")
#    iters = 0
#    outputImage = np.zeros(buffers.memoryBuffer.shape[1:4],dtype = buffers.memoryBuffer.dtype)
#    while iters < 1000:
#         time1 = time.time()
#         outputImage = buffers.readData(outputImage)      
#         print(f"reader进程{os.getpid()}从queue中获取到iters = {iters}, dataTime = {time.time() - time1},带宽v = { (1024) / (time.time() - time1)} MBps")   
#         iters += 1 

# def subProcess(v,lock):
#     for i in range(200):
#         lock.acquire()
#         print(f"子进程{os.getpid()}开始 v.value = {v.value}")
#         v.value += 1 
#         lock.release()

if __name__ == "__main__":
    print(f"主进程({os.getpid()}) begin..")
    width = 1024
    height = 1024
    channels = 1024
    bufferSize = 5
    path = "test2"
    inputImage = np.zeros((height,width,channels),dtype=np.uint8)
    buffers = BufferModule(path,width,height,channels,inputImage.dtype,inputImage.nbytes,bufferSize)
    p2 = Process(target=send,args = (buffers,inputImage,))
    p1 = Process(target=recv,args = (buffers,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    buffers.unlinkShm()
    del buffers
    print(f"主进程({os.getpid()}结束)")
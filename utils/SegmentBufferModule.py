import multiprocessing
from multiprocessing import Process,Lock,shared_memory,Semaphore
import os,time
import numpy as np
import cv2

class SegmentBufferModule:
    def __init__(self,names,bufferSize):
        #memory ID   bufferqueue + Rear + Front + bool
        self.bufferSize = bufferSize
        self.bufferMemorySizes = bufferSize + 2 * 4 + 1 + 4
        self.name = names

        try:
            self.shm = shared_memory.SharedMemory(name = names, create=True,size = self.bufferMemorySizes )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name = names, create=False,size = self.bufferMemorySizes)
            self.shm.unlink()
            self.shm = shared_memory.SharedMemory(name = names, create=True,size = self.bufferMemorySizes )
            #segmentMemory
        self.memorySegmentBuffer = np.ndarray(self.bufferMemorySizes , dtype=np.uint8, buffer=self.shm.buf)

        #读者写者问题的三个锁
        self.lockMutex = Lock()
        self.lockReader = Semaphore(0)
        self.lockWriter = Semaphore(1)

        #queue top
        self.front = np.ndarray((1) , dtype = np.uint32 ,buffer=self.memorySegmentBuffer.data,offset=4)
        self.rear  = np.ndarray((1) , dtype = np.uint32 ,buffer=self.memorySegmentBuffer.data,offset=0)
        self.front[0] = 0
        self.rear[0] = 0
        self.completeReadVariable = np.ndarray((1),dtype=bool,buffer=self.memorySegmentBuffer.data,offset=8)
        self.completeWriteVariable = np.ndarray((1),dtype=bool,buffer=self.memorySegmentBuffer.data,offset=9)
        self.canReadVariable = np.ndarray((1),dtype=bool,buffer=self.memorySegmentBuffer.data,offset=10)
        self.canWriteVariable = np.ndarray((1),dtype=bool,buffer=self.memorySegmentBuffer.data,offset = 11)

        self.canReadVariable[0] = False
        self.completeReadVariable[0] = False
        self.canWriteVariable[0] = True
        self.completeWriteVariable[0] = False

    def canRead(self,dataNbytes):
        return (self.rear[0] >= self.front[0] + dataNbytes)

    def canWrite(self,dataNbytes):
        return (self.rear[0] + dataNbytes <= self.bufferSize)

    def writeData(self,inputData,lengthofData):
        dataNbytes = lengthofData * SegmentBufferModule.getshmBuffer(inputData.dtype)
        while(self.canWrite(dataNbytes) == False):
            self.lockWriter.acquire()
        self.lockMutex.acquire()
        dataBuffer = np.ndarray(lengthofData, dtype=inputData.dtype , buffer = self.memorySegmentBuffer.data,offset = 12 + self.rear[0])
        dataBuffer[:] = inputData[:]
        self.rear[0] = self.rear[0] + dataNbytes
        if(self.rear[0] >= self.bufferSize):
            self.completeWriteVariable[0] = True
            self.rear[0] = self.bufferSize
        if(self.canReadVariable == False):
            self.canReadVariable[0] = True
            self.lockMutex.release()
            self.lockReader.release()
        else:
            self.lockMutex.release()
        #print("1 ",t2-t1,"2 ",t3-t2,"3 ",t4-t3,"4 ",t5-t4,"5 ",t6-t5)


    def readData(self,lengthofData,dtypes):
        t1 = time.time()
        dataNbytes = lengthofData * SegmentBufferModule.getshmBuffer(dtypes)
        while(self.canRead(dataNbytes) == False):
            self.canReadVariable[0] = False
            self.lockReader.acquire()
        self.lockMutex.acquire()
        t2 = time.time()
        dataBuffer = np.ndarray(lengthofData, dtype=dtypes, buffer = self.memorySegmentBuffer.data,offset = 12 + self.front[0]).copy()
        t3 = time.time()
        self.front[0] = self.front[0] + dataNbytes
        if(self.front[0] >= self.bufferSize):
            self.front[0] = 0
            if(self.completeWriteVariable == True):
                self.rear[0] = 0
                self.completeWriteVariable[0] = False
                self.lockMutex.release()
                self.lockWriter.release()
        else:
            self.lockMutex.release()
        t4 = time.time()
        print(t3-t2 ,t4-t1-t3+t2,6220800 / (t4-t1) / 1000 / 1000)
        return dataBuffer

    def getRear(self):
        return self.rear[0]

    def getFront(self):
        return self.front[0]

    def unlinkShm(self):
        self.shm.unlink()

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
        elif (dtype == np.bool):
            return 1
        else:
            raise NotImplementedError

def send(iter,imgs,SBM):
    for j in range(iter):
        for i in range(5):
            lins = imgs[i].nbytes//SegmentBufferModule.getshmBuffer(imgs[i].dtype)
            imgs[i] = imgs[i].reshape(lins)
            SBM.writeData(imgs[i],(lins))

def recv(iter,SBM):
    times = 0
    for j in range(iter):
        b= []
        t2 = time.time()
        for i in range(5):
            lins = imgs[i].nbytes//SegmentBufferModule.getshmBuffer(imgs[i].dtype)
            a = SBM.readData(lins,imgs[i].dtype)
            b.append(a.reshape(1080,1920,3))
        t3 = time.time()
        times = t3 - t2 + times
        for k in range(5):
            cv2.imshow("a",b[k])
            cv2.waitKey()
    # print(times/iter,6220800 * 0.001 * 0.001/(times*0.2/iter))

if __name__ == "__main__":
    SBM = SegmentBufferModule("test2",bufferSize=6220800 * 10)
    imgs = []
    imgs = [cv2.imread("/home/pku/view_synthesis/software/multiProcess/datas/imgs/"+str(i+1)+"-1.png") for i in range(5)]
    b= []
    times = [0,0,0,0]
    iter = 400
    p1 = multiprocessing.Process(target=send,args=(iter,imgs,SBM))
    p1.start()
    p2 = multiprocessing.Process(target=recv,args=(iter,SBM))
    p2.start()
    p1.join()
    p2.join()

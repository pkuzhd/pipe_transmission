from SegmentBufferManager import SegmentBufferManager
from ImageReceiver import ImageData, ImageReceiver
import numpy as np
import cv2
from SegmentBufferModule import SegmentBufferModule
import time
class SegmentReceiver:
    bufferManager : SegmentBufferManager
    
    def __init__(self,SegmentBufferManager,Pipename):
        self.bufferManager = SegmentBufferManager
        self.pipeReceiver = ImageReceiver()
        self.Pipename = Pipename
        self.nameA = self.bufferManager.getNameA()
        self.nameB = self.bufferManager.getNameB()

    def recvfromPipe(self,test_num):
        #input:
        # nbytes + N + imgs[:]
        # imgs = H + W + C + dtype + datas
        # img = [cv2.imread("/home/pku/view_synthesis/software/multiProcess/datas/imgs/"+str(i+1)+"-1.png") for i in range(5)]
        i = 0
        self.pipeReceiver.open(self.Pipename)
        while i < test_num:
            i = i + 1
            #pipeData = ImageData(5,img.copy())
            pipeData = self.pipeReceiver.getData()
            metaData = np.array([4,pipeData.N],dtype=np.uint32)
            #self.bufferManager.recv(self.nameA,nums,1)
            imgs,metaData = SegmentReceiver.encodingImage(pipeData.imgs,metaData)
            self.bufferManager.recv(self.nameA,metaData.astype(np.uint32),metaData.shape[0])
            self.bufferManager.recv(self.nameA,imgs,imgs.shape[0])
            
    def encodingImage(imgs,metaData):
        for iter in range(metaData[1]):
            # w h channels dtype
            imgLength =  imgs[iter].nbytes//SegmentBufferModule.getshmBuffer(imgs[iter].dtype)
            metaData[0] = metaData[0] + imgs[iter].nbytes + 4 * 4 
            metaData = np.concatenate([metaData,imgs[iter].shape,[SegmentBufferModule.setdtype(imgs[iter].dtype)]])
            imgs[iter] = imgs[iter].reshape(imgLength)
        return np.concatenate(imgs,axis=0),metaData  
            
    def encodingDepth(imgs,metaData):
        for iter in range(metaData[1]):
            # w h channels dtype
            metaData[0] = metaData[0] + imgs[iter].nbytes + 4 * 4
            metaData = np.concatenate([metaData,imgs[iter].shape,[SegmentBufferModule.setdtype(imgs[iter].dtype)]])
        imgLengths =  imgs.nbytes//SegmentBufferModule.getshmBuffer(imgs.dtype)
        imgs = imgs.reshape(imgLengths)
        return imgs,metaData                 
                
if __name__ == "__main__":
    name = "../pipe_dir/pipe1"
    bufferNameA = "SBMA"
    bufferNameB = "SBMB"
    bufferSize= 6220800 * 5 * 420 * 1
    
    SBMs = SegmentBufferManager(bufferNameA,bufferNameB,bufferSize,bufferSize)
    Receiver = SegmentReceiver(SBMs,name)
    Receiver.recvfromPipe()

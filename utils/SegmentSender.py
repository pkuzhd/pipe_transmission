from SegmentBufferManager import SegmentBufferManager
from ImageReceiver import ImageData, ImageReceiver
import numpy as np
class SegmentRecviver:
    def __init__(self,SegmentBufferManagerA,Pipename):
        self.bufferManager = SegmentBufferManagerA
        self.pipeReceiver = ImageReceiver()
        self.pipeReceiver.open(Pipename)
        self.nameA = self.bufferManager.getNameA()
        self.nameB = self.bufferManager.getNameB()

    def recvfromPipe(self):
        #input:
        # nbytes + N + imgs[:]
        # imgs = H + W + C + dtype + datas
        while True:
            pipeData = self.pipeReceiver.getData()
            nums = np.array(pipeData.N,dtype=np.uint8)
            nbytes = 0
            #self.bufferManager.recv(self.nameA,nums,1)
            imgs = pipeData.imgs.copy()
            for iter in range(nums[0]):
                nbytes

if __name__ == "__main__":
    name = "../pipe_dir/pipe1"
    a = ImageReceiver()
    a.open(name)
    b = a.getData()
    c = b.imgs
    nbytes = np.zeros(1,dtype=np.uint8)
    n = np.array((0,b.N),dtype=np.uint8)
    for i in range(5):
        nbytes = nbytes + c[i].nbytes


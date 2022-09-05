from SegmentBufferModule import SegmentBufferModule
class SegmentBufferManager():
    def __init__(self,nameA,nameB,bufferSizeA,buffersizeB):
        self.SBMA : SegmentBufferModule
        self.SBMB : SegmentBufferModule

        self.nameA = nameA
        self.nameB = nameB

        self.SBMA = SegmentBufferModule(nameA,bufferSizeA)
        self.SBMB = SegmentBufferModule(nameB,buffersizeB)

    def recv(self,name,inputData,lengthofData):
        if name == self.nameA:
            self.SBMA.writeData(inputData,lengthofData)
        elif name == self.nameB:
            self.SBMB.writeData(inputData,lengthofData)

    def send(self,name,lengthofData,dtypes):
        if name == self.nameA:
            return self.SBMA.readData(lengthofData,dtypes)
        elif name == self.nameB:
            return self.SBMB.readData(lengthofData,dtypes)

    def getNameA(self):
        return self.nameA

    def getNameB(self):
        return self.nameB  

    def unlinks(self):
        self.SBMA.unlinkShm()
        self.SBMB.unlinkShm()
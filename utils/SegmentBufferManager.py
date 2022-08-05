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
        if name == "SBMA":
            self.SBMA.writeData(inputData,lengthofData)
        elif name == "SBMB":
            self.SBMB.writeData(inputData,lengthofData)

    def send(self,name,lengthofData,dtypes):
        if name == "SBMA":
            return self.SBMA.readData(lengthofData,dtypes)
        elif name == "SBMB":
            return self.SBMB.readData(lengthofData,dtypes)

    def getNameA(self):
        return self.nameA

    def getNameB(self):
        return self.nameB
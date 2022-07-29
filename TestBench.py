# from pipe_transmission.utils.MultiProcessBuffer import MultiProcessBuffer
import sys
sys.path.append("/home/pku/view_synthesis/software/multiProcess/pipe_transmission/utils/")
sys.path.append("/home/pku/view_synthesis/software/multiProcess/pipe_transmission/depth_estimation")
from RecvDataProcess import RecvDataProcess
from MultiProcessBuffer import MultiProcessBuffer
from SendDataProcess import SendDataProcess
from DepthProcess import DepthProcess
from depth_estimation.data_preparation import load_cam_paras, scale_camera
from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
import cv2
import os
import numpy as np
import multiprocessing
import torch





if __name__ == "__main__":
    
    print(f"main process {os.getpid()} begin..")
    #data initial
    workspace =  "/home/pku/view_synthesis/software/multiProcess/datas/"
    nameRGB = "bufferRGB4"
    nameDepth = "bufferDepth4"
    nameMask = "bufferMask4"
    nameCrop = "bufferCrop4"
    nameView = "bufferView4"

    bufferSize = 5

    widthRGB = 1920
    heightRGB = 1080
    channelsRGB = 3
    RGBdtypes = np.uint8
    inputRGB = np.zeros((heightRGB,widthRGB,channelsRGB),dtype=RGBdtypes)
    RGBnbytes = inputRGB.nbytes

    widthDepth = 768
    heightDepth = 896
    channelsDepth = 1
    Depthdtypes = np.float32
    inputDepth = np.zeros((heightDepth,widthDepth,channelsDepth),dtype=Depthdtypes)
    Depthnbytes = inputDepth.nbytes

    widthMask = 768
    heightMask = 896
    channelsMask = 1
    Maskdtypes = np.uint8
    inputMask = np.zeros((heightMask,widthMask,channelsMask),dtype=Maskdtypes)
    Masknbytes = inputMask.nbytes

    widthCrop = 0
    heightCrop = 0
    channelsCrop = 4
    Cropdtype = np.uint32
    inputCrop = np.zeros((channelsCrop),dtype = Cropdtype)
    Cropnbytes = inputCrop.nbytes
    
    widthView = 0
    heightView = 0
    channelsView = 1
    Viewdtype = np.uint8
    inputView = np.zeros(channelsView,dtype=Viewdtype)
    Viewnbytes = inputView.nbytes

    bgrs = [cv2.imread(workspace+"background/"+str(i + 1)+".png") for i in range(5)]
    cam_paths = [workspace+"cam_paras/0000000"+str(i)+"_cam.txt" for i in range(5)]
    cams = [load_cam_paras(open(cam_paths[i]), num_depth=32, interval_scale=0.26) for i in range(5)]
    cams = [scale_camera(cams[i], scale=0.5) for i in range(5)]
    matting_model_path = workspace + "TorchScript/torchscript_resnet50_fp32.pth"
    fmn_model_path = workspace +"outputs/pretrained.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    
    #MPB
    MPBs = MultiProcessBuffer(nameRGB,nameDepth,nameMask,widthRGB,heightRGB,channelsRGB,RGBdtypes,RGBnbytes,widthDepth,heightDepth,channelsDepth,\
                Depthdtypes,Depthnbytes,widthMask,heightMask,channelsMask,Maskdtypes,Masknbytes,\
                nameCrop,widthCrop,heightCrop,channelsCrop,Cropdtype,Cropnbytes,\
                nameView,widthView,heightView,channelsView,Viewdtype,Viewnbytes,bufferSize)
    
    #bufferClass
    sendinBufferClass = SendDataProcess(MPBs,workspace)
    recvinBufferClass = RecvDataProcess(MPBs)
    
    #pipes input
    pipetopythonProcess = multiprocessing.Process(target = sendinBufferClass.runRGB,args = ())
    #send to pipes
    pythontopipeProcess = multiprocessing.Process(target=recvinBufferClass.runSender,args =())
    #depth 
    DepthEstimations = DepthEstimation_forRGBD(5, bgrs, cams, matting_model_path, fmn_model_path, device)
    depthProcess = DepthProcess(recvinBufferClass,sendinBufferClass,DepthEstimations)
    pipetopythonProcess.start()
    pythontopipeProcess.start()
    depthProcess.run()
    #depthEstimationProcess = multiprocessing.Process(target=depthProcess.run,args=())
    #pipetopythonProcess.join()
    
    #depthEstimationProcess.start()
    
    
    #depthEstimationProcess.join()
    
    
    # psendRGB = multiprocessing.Process(target = sendRGB,args = (MPB,inputRGB))
    # psendDepth = multiprocessing.Process(target = sendDepth,args = (MPB,inputDepth))
    # psendMask = multiprocessing.Process(target = sendMask,args = (MPB,inputMask))
    
    # preadRGB = multiprocessing.Process(target = recvRGB,args = (MPB,))
    # preadDepth = multiprocessing.Process(target = recvDepth,args = (MPB,))
    # preadMask = multiprocessing.Process(target = recvMask,args = (MPB,))

    # psendRGB.start()
    # preadRGB.start()
    # psendDepth.start()
    # preadDepth.start()
    # psendMask.start()
    # preadMask.start()

    # psendRGB.join()
    # preadRGB.join()
    # psendDepth.join()
    # preadDepth.join()
    # preadMask.join()
    # psendMask.join()

    MPBs.freeRGBBuffer()
    MPBs.freeDepthBuffer()
    MPBs.freeMaskBuffer()
    MPBs.freeCropBuffer()
    MPBs.freeViewBuffer()

    print(f"main process {os.getpid()} end..")
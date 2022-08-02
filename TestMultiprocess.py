import sys
sys.path.append("/home/pku/view_synthesis/software/multiProcess/pipe_transmission/utils/")
sys.path.append("/home/pku/view_synthesis/software/multiProcess/pipe_transmission/depth_estimation")
import cv2
import numpy as np
import torch
import time
import os
import multiprocessing
from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender
from depth_estimation.data_preparation import load_cam_paras, scale_camera
from utils.MultiProcessBuffer import MultiProcessBuffer
from utils.SendDataProcess import SendDataProcess
from utils.RecvDataProcess import RecvDataProcess
from DepthProcess import DepthProcess
#data initial
workspace =  "/home/pku/view_synthesis/software/multiProcess/datas/"
nameRGB = "bufferRGBm"
nameDepth = "bufferDepthm"
nameMask = "bufferMaskm"
nameCrop = "bufferCropm"
nameView = "bufferViewm"

bufferSize = 50

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

imgs = [cv2.imread("/home/pku/view_synthesis/software/multiProcess/datas/imgs/"+str(i+1)+"-1.png") for i in range(5)]
bgrs = [cv2.imread(workspace+"background/"+str(i + 1)+".png") for i in range(5)]
cam_paths = [workspace+"cam_paras/0000000"+str(i)+"_cam.txt" for i in range(5)]
cams = [load_cam_paras(open(cam_paths[i]), num_depth=32, interval_scale=0.26) for i in range(5)]
cams = [scale_camera(cams[i], scale=0.5) for i in range(5)]
matting_model_path = workspace + "TorchScript/torchscript_resnet50_fp32.pth"
fmn_model_path = workspace +"outputs/pretrained.pth"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# initialize backgrond, paras and models
de_rgbd = DepthEstimation_forRGBD(5, bgrs, cams, matting_model_path, fmn_model_path, device)

#MPB
MPBs = MultiProcessBuffer(nameRGB,nameDepth,nameMask,widthRGB,heightRGB,channelsRGB,RGBdtypes,widthDepth,heightDepth,channelsDepth,Depthdtypes,widthMask,heightMask,channelsMask,Maskdtypes,nameCrop,widthCrop,heightCrop,channelsCrop,Cropdtype,nameView,widthView,heightView,channelsView,Viewdtype,bufferSize)

#bufferClass
sendinBufferClass = SendDataProcess(MPBs,'./pipe_dir/pipe1')
recvinBufferClass = RecvDataProcess(MPBs,filenames='./pipe_dir/pipe2')

print("try to open pipe")
# recevier = ImageReceiver()
# recevier.open("./pipe_dir/pipe1")
print("pipe1 open")
#
# sender = RGBDSender()
# sender.open("./pipe_dir/pipe2")
print("pipe2 open")

dirs = './test_dir/'
if not os.path.exists(dirs):
    os.makedirs(dirs)

time_sum = [0, 0, 0, 0]
test_num = 400

depthProcess = DepthProcess(recvinBufferClass,sendinBufferClass,de_rgbd)
pipetopythonProcess = multiprocessing.Process(target = sendinBufferClass.runRGB,args = (imgs,test_num,))
pipetopythonProcess.start()
pythontopipeProcess = multiprocessing.Process(target=recvinBufferClass.runSender,args =(test_num,))
pythontopipeProcess.start()
depthProcess.run(test_num)
    # if (bytes == -1):
    #     print("pipe has been closed.")
    #     sender.close()
    #     break
# print(
#     f"{time_sum[0] / test_num:.3f}, {time_sum[2] / test_num:.3f}, {time_sum[1] / test_num:.3f}, {time_sum[3] / test_num:.3f}")
# print(
#     f"{1 / time_sum[0] * test_num:.1f}, {1 / time_sum[2] * test_num:.1f}, {1 / time_sum[1] * test_num:.1f}, {1 / time_sum[3] * test_num:.1f}")
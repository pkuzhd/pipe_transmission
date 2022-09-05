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
from utils.SegmentBufferManager import SegmentBufferManager
from utils.SegmentSender import SegmentSender
from utils.SegmentReceiver import SegmentReceiver
from SegmentDepthProcess import SegmentDepthProcess


workspace =  "/home/pku/view_synthesis/software/multiProcess/datas/"
bufferA = "bufferA"
bufferB = "bufferB"
bufferSize = 1048576 * 1024

bgrs = [cv2.imread(workspace+"background/"+str(i + 1)+".png") for i in range(5)]
cam_paths = [workspace+"cam_paras/0000000"+str(i)+"_cam.txt" for i in range(5)]
cams = [load_cam_paras(open(cam_paths[i]), num_depth=32, interval_scale=0.26) for i in range(5)]
cams = [scale_camera(cams[i], scale=0.5) for i in range(5)]
matting_model_path = workspace + "TorchScript/torchscript_resnet50_fp32.pth"
fmn_model_path = workspace +"outputs/pretrained.pth"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

de_rgbd = DepthEstimation_forRGBD(5, bgrs, cams, matting_model_path, fmn_model_path, device)

pipeReceiver = './pipe_dir/pipe1'
pipeSender = './pipe_dir/pipe2'
test_num = 400
SBMs = SegmentBufferManager(bufferA,bufferB,bufferSize,bufferSize)
sendModule = SegmentSender(SBMs,pipeSender)
recvModule = SegmentReceiver(SBMs,pipeReceiver)
depthModule = SegmentDepthProcess(sendModule,recvModule,de_rgbd)

#recv
recvProcess = multiprocessing.Process(target = recvModule.recvfromPipe,args=(test_num,))
recvProcess.start()
#send
sendProcess = multiprocessing.Process(target = sendModule.getDepthfromBufferB,args=(test_num,))
sendProcess.start()
#main
depthModule.depthProcess(test_num)
import cv2
import numpy as np
import torch
import time
import os

from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender
from depth_estimation.data_preparation import load_cam_paras, scale_camera

bgrs = [cv2.imread(f"./test_dir/background/{i + 1}.png") for i in range(5)]
cam_paths = [f"./test_dir/cam_paras/0000000{i}_cam.txt" for i in range(5)]
cams = [load_cam_paras(open(cam_paths[i]), num_depth=32, interval_scale=0.26) for i in range(5)]
cams = [scale_camera(cams[i], scale=0.5) for i in range(5)]
matting_model_path = './test_dir/model/torchscript_resnet50_fp32.pth'
fmn_model_path = './test_dir/model/pretrained.pth'

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# initialize backgrond, paras and models
de_rgbd = DepthEstimation_forRGBD(5, bgrs, cams, matting_model_path, fmn_model_path, device)

print("try to open pipe")
recevier = ImageReceiver()
recevier.open("./pipe_dir/pipe1")
print("pipe1 open")
#
sender = RGBDSender()
sender.open("./pipe_dir/pipe2")
print("pipe2 open")

dirs = './test_dir/'
if not os.path.exists(dirs):
    os.makedirs(dirs)

time_sum = [0, 0, 0, 0]
test_num = 400

for j in range(test_num):
    t1 = time.time()
    data = recevier.getData()
    t2 = time.time()
    rgbd_data = de_rgbd.getRGBD(data, crop=True)
    rgbd_data = RGBDData(
        rgbd_data["num_view"],
        rgbd_data["imgs"],
        rgbd_data["depths"],
        rgbd_data["masks"],
        rgbd_data["crops"]
    )
    t3 = time.time()
    bytes = sender.sendData(rgbd_data)

    if (bytes == -1):
        print("pipe has been closed.")
        sender.close()
        break

    t4 = time.time()
    print(j,
          f"get time: {t2 - t1:.3f}, depth estimation time: {t3 - t2:.3f}, send time: {t4 - t3}, total time: {t4 - t1:.3f}")
    time_sum[0] += t2 - t1
    time_sum[1] += t3 - t2
    time_sum[2] += t4 - t3
    time_sum[3] += t4 - t1
print(
    f"{time_sum[0] / test_num:.3f}, {time_sum[1] / test_num:.3f}, {time_sum[2] / test_num:.3f}, {time_sum[3] / test_num:.3f}")
print(
    f"{1 / time_sum[0] * test_num:.1f}, {1 / time_sum[1] * test_num:.1f}, {1 / time_sum[2] * test_num:.1f}, {1 / time_sum[3] * test_num:.1f}")
recevier.close()
sender.close()

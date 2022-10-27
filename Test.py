from asyncio import FastChildWatcher
import cv2
import numpy as np
import torch
import time
import os

from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender
from depth_estimation.data_preparation import load_cam_paras, scale_camera

# imgs = [cv2.imread(f"/data/GoPro/videos/teaRoom/sequence/1080p/video/{i+1}-102.png") for i in range(5)]
# # imgs = [cv2.resize(cv2.imread(f"/data/GoPro/videos/teaRoom/frames/{i+1}.1-14.png"), None, fx=1/2, fy=1/2) for i in range(5)]
# bgrs = [cv2.imread(f"/data/GoPro/videos/teaRoom/sequence/1080p/background/{i + 1}.png") for i in range(5)]
# cam_paths = [f"/home/wph/pipe_transmission/cam_paras/0000000{i}_cam.txt" for i in range(5)]
# cams = [load_cam_paras(open(cam_paths[i]), num_depth=32, interval_scale=0.26) for i in range(5)]
# cams = [scale_camera(cams[i], scale=0.5) for i in range(5)]

imgs = [cv2.imread(f"/data/zhanghaodan/web/demo2/videos/1-{i + 1}.png") for i in range(5)]
bgrs = [cv2.imread(f"/data/zhanghaodan/web/demo2/background/1-{i + 1}.png") for i in range(5)]
cam_paths = [f"/data/zhanghaodan/web/para/0000000{i}_cam.txt" for i in range(5)]
cams = [load_cam_paras(open(cam_paths[i]), num_depth=32, interval_scale=0.26) for i in range(5)]

matting_model_path = '/home/wph/BackgroundMatting/TorchScript/torchscript_resnet50_fp32.pth'
fmn_model_path = '/home/wph/FastMVSNet/outputs/pretrained.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# initialize backgrond, paras and models
de_rgbd = DepthEstimation_forRGBD(5, bgrs, cams, matting_model_path, fmn_model_path, device)

# GN CONSIST PROPAGA CONSIST
dirs = '/data/pipe_depth/wph/1027_demo2/demo2/'
WRITE = 1
BYTES = 0

if not os.path.exists(dirs):
    print("create new file")
    os.makedirs(dirs)

for i in range(1):
    # cur_imgs = np.stack(imgs)
    data = ImageData(5, imgs)
    st = time.time()

    rgbd_data = de_rgbd.getRGBD_test(data, crop=(i==0), isGN=False, checkConsistancy=True, propagation=True, checkSecondConsistancy=True, show_depth=False)

    print(f"total time: {time.time() - st}")

    for key in rgbd_data.keys():
        if key == 'crops' or key == 'num_view': continue
        print(f"rgbd-{key}: {rgbd_data[key][0].shape} {rgbd_data[key][0].dtype}")

    for key in rgbd_data.keys():
        if key == 'imgs':
            i = 0
            for img in rgbd_data[key]:
                if WRITE: 
                    cv2.imwrite(dirs + f"imgs_{i}.jpg", img)

                i += 1
            print(f"{key} saved...")

        elif key == 'depths':
            i = 0
            for depth in rgbd_data[key]:
                if WRITE:
                    cv2.imwrite(dirs + f"depth_{i}.jpg", depth)
                if BYTES:
                    with open(f"/data/pipe_depth/0920/depth_{i}", 'wb') as f:
                        f.write(depth.tobytes())
                i += 1
            print(f"{key} saved...")


        elif key == 'masks':
            i = 0
            for mask in rgbd_data[key]:
                if WRITE:
                    cv2.imwrite(dirs + f"mask_{i}.jpg", mask)
                if BYTES:
                    with open(f"/data/pipe_depth/0920/mask_{i}", 'wb') as f:
                        f.write(mask.tobytes())
                i += 1
            print(f"{key} saved...")

        elif key == 'crops':
            for crop in rgbd_data[key]:
                if WRITE:
                    with open(f"{dirs}crops.txt", 'a') as f:
                        print(f"rgbd: crop {crop}")
                        f.write(str(crop)+'\n')
            print(f"{key} saved...")

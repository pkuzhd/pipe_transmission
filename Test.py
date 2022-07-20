import cv2
import numpy as np
import torch
import time

from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender
from depth_estimation.data_preparation import load_cam_paras, scale_camera


imgs = [cv2.imread(f"/data/GoPro/videos/teaRoom/sequence/1080p/video/{i+1}-1.png") for i in range(5)]
bgrs = [cv2.imread(f"/data/GoPro/videos/teaRoom/sequence/1080p/background/{i+1}.png") for i in range(5)]
cam_paths = [f"/home/wph/pipe_transmission/cam_paras/0000000{i}_cam.txt" for i in range(5)]
# cams = get_ori_cam_paras(5, cam_paths, num_virtual_plane=32, interval_scale=0.26)
cams = [load_cam_paras(open(cam_paths[i]), num_depth=32, interval_scale=0.26) for i in range(5)]
cams = [scale_camera(cams[i], scale = 0.5) for i in range(5)]

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# initialize backgrond and paras
de_rgbd = DepthEstimation_forRGBD(5, bgrs, cams, device) 

for i in range(1):
    data = ImageData(5, imgs)
    st = time.time()
    rgbd_data = de_rgbd.getRGBD(data, crop=True)
    print(f"total time: {time.time() - st}")
    
    for key in rgbd_data.keys():
        if key == 'crops' or key == 'num_view': continue
        print(f"rgbd-{key}: {rgbd_data[key][0].shape} {rgbd_data[key][0].dtype}")

    for key in rgbd_data.keys():
        if key == 'imgs':
            i = 0
            for img in rgbd_data[key]:
                cv2.imwrite(f"/home/wph/results/1080p_refcrop_resize0.8/imgs_{i}.jpg", (img))
                i += 1
        elif key == 'depths':
            i = 0
            for depth in rgbd_data[key]:
                cv2.imwrite(f"/home/wph/results/1080p_refcrop_resize0.8/depth_{i}.jpg", depth)
                i += 1
        elif key == 'masks':
            i = 0
            for mask in rgbd_data[key]:
                cv2.imwrite(f"/home/wph/results/1080p_refcrop_resize0.8/mask_{i}.jpg", mask)
                i += 1
        elif key == 'crops':
            for crop in rgbd_data[key]:
                print(f"rgbd: crop {crop}")

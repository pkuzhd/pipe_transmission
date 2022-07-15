import cv2
import numpy as np

from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender

# imgs = [cv2.imread(f"./test_data/{j + 1}.fg.jpg") for j in range(5)]
imgs = [cv2.imread(f"/data/FastMVSNet/lab/Rectified/scan1/rect_00{i+1}_3_r5000.png") for i in range(5)]
bgrs = [cv2.imread(f"/data/FastMVSNet/lab/Rectified/scan1_crop_b/rect_00{i+1}_3_r5000.png") for i in range(5)]
cam_paths = [f"/data/FastMVSNet/lab/Cameras/0000000{i}_cam.txt" for i in range(5)]

# initialize backgrond and paras
de_rgbd = DepthEstimation_forRGBD(5, bgrs, cam_paths) 

for i in range(1):
    data = ImageData(5, imgs)
    rgbd_data = de_rgbd.getRGBD(data)
    
    for key in rgbd_data.keys():
        if key == 'crops' or key == 'num_view': continue
        print(f"rgbd-{key}: {rgbd_data[key][0].shape} {rgbd_data[key][0].dtype}")

    for key in rgbd_data.keys():
        if key == 'imgs':
            i = 0
            for img in rgbd_data[key]:
                cv2.imwrite(f"/home/wph/pipe_transmission/depth_estimation/0715results/imgs_{i}.jpg", (img))
                i += 1
        elif key == 'depths':
            i = 0
            for depth in rgbd_data[key]:
                cv2.imwrite(f"/home/wph/pipe_transmission/depth_estimation/0715results/depth_{i}.jpg", depth)
                i += 1
        elif key == 'masks':
            i = 0
            for mask in rgbd_data[key]:
                cv2.imwrite(f"/home/wph/pipe_transmission/depth_estimation/0715results/mask_{i}.jpg", mask)
                i += 1
        elif key == 'crops':
            for crop in rgbd_data[key]:
                print(f"rgbd: crop {crop}")

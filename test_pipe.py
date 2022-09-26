import cv2
import numpy as np
import torch
import time
import os

from depth_estimation.DepthEstimation import DepthEstimation_forRGBD
from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender
from depth_estimation.data_preparation import load_cam_paras, scale_camera

st = time.time()
imgs = []
for scan in range(12,33):
    # img = [cv2.imread(f"/data/GoPro/videos/teaRoom/sequence/1080p/video/{i+1}-{scan+1}.png") for i in range(5)]
    img = [cv2.resize(cv2.imread(f"/data/GoPro/videos/teaRoom/frames/{i+1}.1-{scan}.png"), None, fx=0.5, fy=0.5) for i in range(5)]
    imgs.append(img)

bgrs = [cv2.imread(f"/data/GoPro/videos/teaRoom/sequence/1080p/background/{i+1}.png") for i in range(5)]
cam_paths = [f"/home/wph/pipe_transmission/cam_paras/0000000{i}_cam.txt" for i in range(5)]
cams = [load_cam_paras(open(cam_paths[i]), num_depth=32, interval_scale=0.26) for i in range(5)]
cams = [scale_camera(cams[i], scale = 0.5) for i in range(5)]

matting_model_path = '/home/wph/BackgroundMatting/TorchScript/torchscript_resnet50_fp32.pth'
fmn_model_path = '/home/wph/FastMVSNet/outputs/pretrained.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
output_path = '/data/pipe_depth/1080p_autocrop_jql/'

WRITE = 1
BYTES = 0

# initialize backgrond and paras
de_rgbd = DepthEstimation_forRGBD(5, bgrs, cams, matting_model_path, fmn_model_path, device) 
et = time.time()
print(f"Initializing done...time: {(et-st)}s")

for i in range(4):
    # cur_imgs = np.stack(imgs[i])
    data = ImageData(5, imgs[i])
    rgbd_data = de_rgbd.getRGBD_test(data, crop=(i%5==0), isGN=False, checkConsistancy=True, propagation=True, checkSecondConsistancy=True)

total_t = 0
st = time.time()
for i in range(0, 21):
    # cur_imgs = np.stack(imgs[i])
    st = time.time()
    data = ImageData(5, imgs[i])
    rgbd_data = de_rgbd.getRGBD_test(data, crop=(i%5==0), isGN=False, checkConsistancy=True, propagation=True, checkSecondConsistancy=True)
    et = time.time()
    total_t += et - st
    
    dirs = output_path + f'scan{12+i}/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # for key in rgbd_data.keys():
    #     if key == 'crops' or key == 'num_view': continue
    #     print(f"rgbd-{key}: {rgbd_data[key][0].shape} {rgbd_data[key][0].dtype}")

    for key in rgbd_data.keys():
        if key == 'imgs':
            i = 0
            for img in rgbd_data[key]:
                if WRITE: 
                    cv2.imwrite(dirs + f"imgs_{i}.jpg", img)

                i += 1

        elif key == 'depths':
            i = 0
            for depth in rgbd_data[key]:
                if WRITE:
                    cv2.imwrite(dirs + f"depth_{i}.jpg", depth)
                if BYTES:
                    with open(f"/data/pipe_depth/0920/depth_{i}", 'wb') as f:
                        f.write(depth.tobytes())
                i += 1


        elif key == 'masks':
            i = 0
            for mask in rgbd_data[key]:
                if WRITE:
                    cv2.imwrite(dirs + f"mask_{i}.jpg", mask)
                if BYTES:
                    with open(f"/data/pipe_depth/0920/mask_{i}", 'wb') as f:
                        f.write(mask.tobytes())
                i += 1

        elif key == 'crops':
            for crop in rgbd_data[key]:
                if WRITE:
                    with open(f"{dirs}crops.txt", 'a') as f:
                        f.write(str(crop)+'\n')

et = time.time()
# total_t = et-st
print(f"total time: {total_t} avg time: {total_t/100}")
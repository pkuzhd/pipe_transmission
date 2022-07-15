#!/usr/bin/env python
import argparse
import os.path as osp
import logging
import time
import sys
import cv2
from cv2 import INTER_LINEAR
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, osp.dirname(__file__) + '/..')

from fastmvsnet.utils.logger import setup_logger
from fastmvsnet.config import load_cfg_from_file
from fastmvsnet.model_sf import FastMVSNet_singleframe
from fastmvsnet.data_preparation_if import build_data_forFast, get_ori_cam_paras
from fastmvsnet.utils.checkpoint import Checkpointer

def get_fore_rect(back, src):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=100, detectShadows=False)
    
    # get the front mask
    mask = fgbg.apply(back)
    mask = fgbg.apply(src)

    # eliminate the noise
    # line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), (-1, -1))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)

    max_area = -1
    max_rect = None
    # find the max area contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area < 150:
            continue
        rect = cv2.minAreaRect(contours[c])

        if area > max_area:
            max_area = area
            max_rect = rect

    box = cv2.boxPoints(max_rect)
    max_x, min_x = np.max(box[:,0]), np.min(box[:,0])
    max_y, min_y = np.max(box[:,1]), np.min(box[:,1])
    box = np.array([[min_x, max_y],[min_x, min_y],[max_x, min_y],[max_x, max_y]])
    box = np.int0(box)

    # min_row = int(box[1][1]/4) *4
    # max_row = int((box[0][1]+3)/4) *4
    # min_col = int(box[0][0]/4) *4
    # max_col = int((box[2][0]+3)/4) *4

    max_x, max_y = int((max_x+3)/4) *4, int((max_y+3)/4) *4
    min_x, min_y = int(min_x/4) *4, int(min_y/4) *4

    w = max_x - min_x
    h = max_y - min_y

    return (w, h, min_x, min_y)


# 读cfg
cfg = load_cfg_from_file('/home/wph/pipe_transmission/depth_estimation/FastMVSNet/configs/single_frame.yaml')
# self.cfg.merge_from_list(args.opts)
cfg.freeze()


# 初始化FastMVSNet，读background， cam_paras
FastMVSNet_model = FastMVSNet_singleframe(
    img_base_channels=cfg.MODEL.IMG_BASE_CHANNELS,
    vol_base_channels=cfg.MODEL.VOL_BASE_CHANNELS,
    flow_channels=cfg.MODEL.FLOW_CHANNELS
    )

FastMVSNet_model = nn.DataParallel(FastMVSNet_model).cuda()
stat_dict = torch.load(cfg.TEST.WEIGHT, map_location=torch.device("cpu"))
FastMVSNet_model.load_state_dict(stat_dict.pop("model"), strict = False)
# print(f"! Model: {FastMVSNet_model}")

cam_paths = [f"/data/FastMVSNet/lab/Cameras/0000000{i}_cam.txt" for i in range(5)]
cams = get_ori_cam_paras(5, cam_paths, num_virtual_plane=cfg.DATA.TEST.NUM_VIRTUAL_PLANE, interval_scale=cfg.DATA.TEST.INTER_SCALE)
# print(f"# cams: {len(cams)}, {cams[0].shape}")
bgrs = [cv2.imread(f"/home/wph/pipe_transmission/test_data/{i+1}.jpg") for i in range(5)]

# 准备数据，crop
imgs = [cv2.imread(f"/data/FastMVSNet/lab/Rectified/scan1/rect_00{i+1}_3_r5000.png") for i in range(5)]
# crops = [get_fore_rect(imgs[i], bgrs[i]) for i in range(5)]
# cropped_imgs =[imgs[i][crops[i][3]:crops[i][3]+crops[i][1], crops[i][2]:crops[i][2]+crops[i][0], :] for i in range(5)] 


# 预处理img，cams
print(f"! testDepth_if: {imgs[0].shape}")
imgs_tensor, cams_tensor = build_data_forFast(imgs, cams, height=cfg.DATA.TEST.IMG_HEIGHT, width=cfg.DATA.TEST.IMG_WIDTH)
imgs_tensor, cams_tensor = imgs_tensor.cuda(non_blocking=True), cams_tensor.cuda(non_blocking=True)

# 跑model
# print(F"cfg: {cfg.MODEL.TEST.IMG_SCALES}, {cfg.MODEL.TEST.INTER_SCALES}")
with torch.no_grad():
    preds = FastMVSNet_model(imgs_tensor = imgs_tensor, cams_tensor = cams_tensor, img_scales=cfg.MODEL.TEST.IMG_SCALES, inter_scales=cfg.MODEL.TEST.INTER_SCALES, blending=None, isGN=False, isTest=True)

    for key in preds.keys():
        print(f"! get key of {key}")
        if "coarse_depth_map" not in key:
            continue
        print(f"! preds: {preds[key].shape}")
        tmp = preds[key][0,0].to(torch.device('cpu')).detach().numpy()
        # tmp = 1.0 / tmp
        maxn = np.max(tmp)#1.6###1.6
        minn = np.min(tmp)#0.7###1.0
        print('max:{},min:{}'.format(maxn, minn))
        tmp = (tmp - minn) / (maxn - minn) * 255.0
        # tmp = tmp.astype('uint8')
        # tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_RAINBOW)
        # cv2.imwrite(f"/home/wph/pipe_transmission/depth_estimation/test_results/preds_{key}_{tmp.shape}_of_{tmp.dtype}.jpg", tmp)
        
        tmp_ori = cv2.resize(tmp,dsize=None, fx=4, fy=4, interpolation=INTER_LINEAR)
        cv2.imwrite(f"/home/wph/pipe_transmission/depth_estimation/test_results/depths/{key}_0715.jpg", tmp_ori)



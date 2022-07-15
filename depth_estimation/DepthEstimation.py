import os
import sys
from matplotlib import image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__) + '/..')
# print(f"! {os.path.dirname(__file__)}")
sys.path.insert(0, '/home/wph/pipe_transmission/depth_estimation/fastmvsnet/..')

from cv2 import INTER_LINEAR
from torchvision.transforms.functional import to_tensor
from fastmvsnet.model_sf import FastMVSNet_singleframe

from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender
from torch import unsqueeze
from fastmvsnet.config import load_cfg_from_file
from data_preparation import build_data_forFast, get_ori_cam_paras


# 根据前后景得到最大聚集所在的矩形(w,h)可被4整除
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

class DepthEstimation_forRGBD():
    # 要求输入img为RGB

    def __init__(self, view_num, backgrounds, cam_paths):
        # paras to read in
        # initialize backgrounds
        self.num_view = view_num
        self.bgrs = backgrounds.copy()

        # Crop: no model

        # Matting: load model
        self.Matting_model = torch.jit.load('/home/wph/BackgroundMatting/TorchScript/torchscript_resnet50_fp32.pth').eval().cuda()

        # Depth estimation: 
        # config folder for cams and 
        self.cfg = load_cfg_from_file('/home/wph/pipe_transmission/depth_estimation/single_frame.yaml')
        self.cfg.freeze()

        # get cameras' paras
        self.cams = get_ori_cam_paras(5, cam_paths, num_virtual_plane=self.cfg.DATA.TEST.NUM_VIRTUAL_PLANE, interval_scale=self.cfg.DATA.TEST.INTER_SCALE)

        # build model
        self.FastMVSNet_model = FastMVSNet_singleframe()
        self.FastMVSNet_model = nn.DataParallel(self.FastMVSNet_model).cuda()
        stat_dict = torch.load('/home/wph/FastMVSNet/outputs/pretrained.pth', map_location=torch.device("cpu"))
        self.FastMVSNet_model.load_state_dict(stat_dict.pop("model"), strict = False)
    
    def get_masks(self, pha):
        masks = []
        pha_np = pha.to(torch.device('cpu')).numpy()

        for i in range(5):
            mask_np = np.transpose(pha_np[i], (1,2,0))
            mask = np.uint8(mask_np*255)
            masks.append(mask)

        return masks

    def get_depths(self, preds):
        depths = []
        for key in preds.keys():
            # print(f"! get key of {key}")
            if "coarse_depth_map" not in key:
                continue
            # print(f"! preds: {preds[key].shape}")
            tmp = preds[key][0,0].to(torch.device('cpu')).detach().numpy()
            # tmp = 1.0 / tmp
            maxn = np.max(tmp)#1.6###1.6
            minn = np.min(tmp)#0.7###1.0
            # print('max:{},min:{}'.format(maxn, minn))
            tmp = (tmp - minn) / (maxn - minn) * 255.0
            # tmp = tmp.astype('uint8')
            # tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_RAINBOW)
            # cv2.imwrite(f"/home/wph/pipe_transmission/depth_estimation/test_results/preds_{key}_{tmp.shape}_of_{tmp.dtype}.jpg", tmp)
            
            tmp_ori = cv2.resize(tmp,dsize=None, fx=4, fy=4, interpolation=INTER_LINEAR)
            depths.append(tmp_ori)
            # cv2.imwrite(f"/home/wph/pipe_transmission/depth_estimation/test_results/depths/{key}_0715.jpg", tmp_ori)
        
        return depths

    def getRGBD(self, imgdata):
        self.imgs = imgdata.imgs.copy()

        # Crop to get crops(w,h,x,y), cropped_imgs
        self.cropped_imgs = []
        self.cropped_bgrs = []
        self.crops = []
        for i in range(self.num_view):
            cur_img = self.imgs[i]
            cur_bgr = self.bgrs[i]
            w, h, x, y = get_fore_rect(cur_bgr, cur_img)    # (w, h, x, y)
            
            self.crops.append((w, h, x, y))
            cropped_img = cur_img[y:y+h, x:x+w, :]
            # print(cropped_img.shape)
            self.cropped_imgs.append(cropped_img)
            cropped_bgr = cur_bgr[y:y+h, x:x+w, :]
            self.cropped_bgrs.append(cropped_bgr)
            # print(cropped_bgr.shape)

        # to tensor to device: cropped_bgrs cropped_imgs
        # use original pics for test
        # matting tensor
        # print(self.cropped_imgs[0].shape)
        # print(self.cropped_bgrs[0].shape)
        imgs_m = np.stack(self.imgs, axis=0)
        bgrs_m = np.stack(self.bgrs, axis=0)
        imgs_tensor_m = torch.tensor(imgs_m).permute(0,3,1,2).float()
        imgs_tensor_m = (imgs_tensor_m/255).cuda()
        bgrs_tensor_m = torch.tensor(bgrs_m).permute(0,3,1,2).float()
        bgrs_tensor_m = (bgrs_tensor_m/255).cuda()

        # depth estimation tensor
        imgs_tensor, cams_tensor = build_data_forFast(self.imgs, self.cams, height=self.cfg.DATA.TEST.IMG_HEIGHT, width=self.cfg.DATA.TEST.IMG_WIDTH)
        imgs_tensor, cams_tensor = imgs_tensor.cuda(non_blocking=True), cams_tensor.cuda(non_blocking=True)  # data_batch

        # run model of depths and masks
        with torch.no_grad():
            preds = self.FastMVSNet_model(imgs_tensor = imgs_tensor, cams_tensor = cams_tensor, img_scales=self.cfg.MODEL.TEST.IMG_SCALES, inter_scales=self.cfg.MODEL.TEST.INTER_SCALES, blending=None, isGN=False, isTest=True)
        
        alpha_masks = self.Matting_model(imgs_tensor_m, bgrs_tensor_m)[0]
        
        # depths/masks: tensor->numpy
        self.depths = self.get_depths(preds)
        self.masks = self.get_masks(alpha_masks)

        return {
            "num_view": self.num_view,
            "imgs": self.imgs,
            "depths": self.depths, 
            "masks": self.masks, 
            "crops": self.crops
        }
        

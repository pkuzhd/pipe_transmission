from email.mime import base
import os
import sys
from matplotlib import image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time

sys.path.insert(0, os.path.dirname(__file__) + '/..')
# print(f"! {os.path.dirname(__file__)}")
sys.path.insert(0, '/home/wph/pipe_transmission/depth_estimation/fastmvsnet/..')

from torchvision.transforms.functional import to_tensor
from fastmvsnet.model_sf import FastMVSNet_singleframe

from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender
from fastmvsnet.config import load_cfg_from_file
from data_preparation import build_data_forFast, build_data_forFast_sc, get_ori_cam_paras, adjust_cam_para, write_cam_dtu, scale_camera
from fastmvsnet.utils.preprocess import  crop_dtu_input, scale_dtu_input


# 根据前后景得到最大聚集所在的矩形(w,h)
def get_fore_rect(back, src):
    maxh = back.shape[0]
    maxw = back.shape[1]
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

    max_x, max_y = int(min(maxw, max_x)), int(min(maxh, max_y))
    min_x, min_y = int(max(min_x, 0)), int(max(min_y, 0))

    w = max_x - min_x
    h = max_y - min_y

    return (w, h, min_x, min_y)

def get_crops(imgs, bgrs, base_size):
    img_h, img_w,_ = imgs[0].shape
    num_view = len(imgs)
    rects = []
    crops = []
    maxw = 0
    maxh = 0

    for i in range(num_view):
        w, h, x, y = get_fore_rect(imgs[i], bgrs[i])
        if w > maxw: maxw = w
        if h > maxh: maxh = h
        rects.append((w, h, x, y))
    
    # fit base_images
    maxw = int(int((maxw + base_size - 1)/base_size) * base_size)
    maxh = int(int((maxh + base_size - 1)/base_size) * base_size)
    if maxw > img_w: maxw = maxw - base_size
    if maxh > img_h: maxh = maxh - base_size

    # get new crops base point
    for i in range(num_view):
        w, h, x, y = rects[i]
        newx, newy = x+w-maxw, y+h-maxh
        if newx < 0: newx = 0
        if newy < 0: newy = 0
        crops.append((maxw, maxh, newx, newy))

    return crops, maxw, maxh

def crop_images(imgs, crops):
    num = len(imgs)
    cropped_imgs = []
    
    for i in range(num):
        w, h, x, y = crops[i]
        cropped_img = imgs[i][y:y+h, x:x+w, :]
        cropped_imgs.append(cropped_img)
    
    return cropped_imgs


def get_masks(pha):
    masks = []
    pha_np = pha.to(torch.device('cpu')).numpy()

    for i in range(5):
        mask_np = np.transpose(pha_np[i], (1,2,0))
        mask = np.uint8(mask_np*255)
        masks.append(mask)

    return masks

def get_depths(preds, alpha):
    depths = []
    for key in preds.keys():
        # print(f"! get key of {key}")
        if "coarse_depth_map" not in key:
            continue
        # print(f"! preds: {preds[key].shape}")
        # print("alpha: ", alpha.shape)
        # print("preds:", preds[key][0,0].shape)
        tmp = preds[key][0,0].to(torch.device('cpu')).detach().numpy()
        # tmp = 1.0 / tmp
        maxn = np.max(tmp)#1.6###1.6
        minn = np.min(tmp)#0.7###1.0
        # maxn = 1.6
        # minn = 0.6
        # print('max:{},min:{}'.format(maxn, minn))
        tmp = (tmp - minn) / (maxn - minn) * 255.0
        # tmp = tmp.astype('uint8')
        # tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_RAINBOW)
        
        tmp_ori = cv2.resize(tmp,dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        # tmp_ori = tmp_ori * (alpha[i][:,:,0]/255)
        
        depths.append(tmp_ori)

    
    return depths

class DepthEstimation_forRGBD():
    # 要求输入img为RGB

    def __init__(self, view_num, backgrounds, cams, device):
        # paras to read in
        # initialize backgrounds
        self.num_view = view_num
        self.device = device
        self.bgrs = backgrounds.copy()
        self.cams = cams.copy()
        self.matting_model_path = '/home/wph/BackgroundMatting/TorchScript/torchscript_resnet50_fp32.pth'
        self.fmn_cfg_path = '/home/wph/pipe_transmission/depth_estimation/single_frame.yaml'
        self.fmn_model_path = '/home/wph/FastMVSNet/outputs/pretrained.pth'

        # Crop: no model
        
        # Matting: load model
        self.Matting_model = torch.jit.load(self.matting_model_path).eval().to(device)

        # build model
        self.FastMVSNet_model = FastMVSNet_singleframe()
        self.FastMVSNet_model = nn.DataParallel(self.FastMVSNet_model).to(self.device)
        # self.FastMVSNet_model = self.FastMVSNet_model.module
        # # self.FastMVSNet_model = self.FastMVSNet_model.eval().to(device)
        stat_dict = torch.load(self.fmn_model_path, map_location=torch.device("cpu"))
        self.FastMVSNet_model.load_state_dict(stat_dict.pop("model"), strict = False)

            


    def getRGBD(self, imgdata, crop=False):
        times = 1
        r_scale = 0.5
        ori_imgs = imgdata.imgs.copy()
        # ref_crop = [(720, 880, 850, 189), (720, 880, 780, 111), (720, 880, 680, 198), (720, 880, 550, 200), (720, 880, 491, 159)]
        # ref_crop = [(768, 896, 850, 170), (768, 896, 780, 111), (768, 896, 680, 170), (768, 896, 550, 180), (768, 896, 491, 159)]
        ref_crop = [(768, 896, 768, 173), (768, 896, 695, 95), (768, 896, 613, 182), (768, 896, 567, 184), (768, 896, 443, 143)]
        cams = [self.cams[i].copy() for i in range(self.num_view)]

        # Crop to get crops(w,h,x,y), cropped_imgs
        # crop_st = time.time()
        if crop:
            # self.crops, maxw, maxh = get_crops(self.imgs, self.bgrs, 64/r_scale)
            crops = ref_crop
            cropped_imgs = crop_images(ori_imgs, crops)
            cropped_bgrs = crop_images(self.bgrs, crops)
            cropped_cams = adjust_cam_para(cams, crops)
        else:
            crops = []
            cropped_imgs = imgs
            cropped_bgrs = bgrs
            cropped_cams = cams
        
        # crop_et = time.time()

        # resize to r_scale for FastMVSNet/Matting
        cropped_imgs= [cv2.resize(cropped_imgs[i], None, fx=r_scale, fy=r_scale) for i in range(5)]
        cropped_bgrs = [cv2.resize(cropped_bgrs[i], None, fx=r_scale, fy=r_scale) for i in range(5)]
        cropped_cams = [scale_camera(cropped_cams[i], scale=r_scale) for i in range(5)]

        # resize_et = time.time()

        # to tensor to device: cropped_bgrs cropped_imgs
        # matting tensor
        imgs = np.stack(cropped_imgs, axis=0)
        bgrs = np.stack(cropped_bgrs, axis=0)
        imgs_tensor = torch.tensor(imgs).permute(0,3,1,2).float().to(self.device)
        bgrs_tensor = torch.tensor(bgrs).permute(0,3,1,2).float().to(self.device)
        imgs_tensor_m = (imgs_tensor/255.0)
        bgrs_tensor_m = (bgrs_tensor/255.0)

        if crop:
            cam_params_list = np.stack(cropped_cams, axis=0)
            cam_params_list = torch.tensor(cam_params_list).float().to(self.device)
            cams_tensor = cam_params_list.unsqueeze(0)
            imgs_tensor = imgs_tensor_m.unsqueeze(0)
        else:
            croped_images, croped_cams = crop_dtu_input(cropped_imgs, cropped_cams,height=imgs[0].shape[0]*r_scale, width=imgs[0].shape[1]*r_scale, base_image_size=64, depth_image=None)
            img_list = np.stack(croped_images, axis=0)
            cam_params_list = np.stack(croped_cams, axis=0)
            img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
            cam_params_list = torch.tensor(cam_params_list).float()

            imgs_tensor = img_list.unsqueeze(0).to(self.device)
            cams_tensor = cam_params_list.unsqueeze(0).to(self.device)

        # trans_et = time.time()

        with torch.no_grad():
            # for i in range(3):
            #     alpha_masks = self.Matting_model(imgs_tensor_m, bgrs_tensor_m)[0]
            
            # torch.cuda.synchronize()
            # matting_st = time.time()
            
            for i in range(times):
                alpha_masks = self.Matting_model(imgs_tensor_m, bgrs_tensor_m)[0]

            # torch.cuda.synchronize()
            # mattint_et = time.time()

        # run model of depths and masks
        with torch.no_grad():
            # for i in range(3):
            #     preds = self.FastMVSNet_model(imgs_tensor = imgs_tensor, cams_tensor = cams_tensor, img_scales=(0.25, 0.5, 1.0), inter_scales=(0.75, 0.15, 0.15), blending=None, isGN=False, isTest=True)

            # torch.cuda.synchronize()
            # depth_st = time.time()
            for i in range(times):
                preds = self.FastMVSNet_model(imgs_tensor = imgs_tensor, cams_tensor = cams_tensor, img_scales=(0.25, 0.5, 1.0), inter_scales=(0.75, 0.15, 0.15), blending=None, isGN=False, isTest=True)
            # torch.cuda.synchronize()
            # depth_et = time.time()
        
        # depths/masks: tensor->numpy
        masks = get_masks(alpha_masks)
        depths = get_depths(preds, masks)

        # get_back_et = time.time()

        masks= [cv2.resize(masks[i], None, fx=1/r_scale, fy=1/r_scale) for i in range(5)]
        depths = [cv2.resize(depths[i], None, fx=1/r_scale, fy=1/r_scale) for i in range(5)]

        # et = time.time()

        # print(f"crop time: {crop_et - crop_st}s")
        # print(f"resize time: {resize_et - crop_et}s")
        # print(f"trans time: {trans_et - resize_et}s")
        # print(f"matting time: {(mattint_et - matting_st)/times}s")
        # print(f"depth_estimation time: {(depth_et - depth_st)/times}s")
        # print(f"trans and resize: {get_back_et - depth_et}s")
        # print(f"trans and resize: {et - get_back_et}s")


        # 一致性校验
        # 扩散

        return {
            "num_view": self.num_view,
            "imgs": ori_imgs,
            "depths": depths, 
            "masks": masks, 
            "crops": crops
        }
        

from email.mime import base
import os
import sys
from cv2 import IMWRITE_PNG_STRATEGY_DEFAULT
from matplotlib import image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time
from torchvision import transforms 
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__) + '/..')
# print(f"! {os.path.dirname(__file__)}")
sys.path.insert(0, './depth_estimation')

from torchvision.transforms.functional import to_tensor
from fastmvsnet.model_sf import FastMVSNet_singleframe

from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender
from fastmvsnet.config import load_cfg_from_file
from data_preparation import build_data_forFast, build_data_forFast_sc, get_ori_cam_paras, adjust_cam_para, write_cam_dtu, scale_camera
from fastmvsnet.utils.preprocess import  crop_dtu_input, scale_dtu_input
from picutils import ConsistancyChecker, MyPerspectiveCamera


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
    """
    imgs: (list) H*W
    crops: (w,h,x,y)
    """
    num = len(imgs)
    cropped_imgs = []
    
    for i in range(num):
        w, h, x, y = crops[i]
        cropped_img = imgs[i][y:y+h, x:x+w, :]
        cropped_imgs.append(cropped_img)
    
    return cropped_imgs

def crop_bgrs_tensor(imgs_tensor, crops, device):
    """
    imgs_tensor: (tensor) V*3*H*W
    return:(tensor) V*3*crop_h*crop_w
    """
    num = imgs_tensor.shape[0]
    width, height = crops[0][0], crops[0][1]
    cropped_imgs = torch.ones((num, 3, height, width)).to(device)
    
    for i in range(num):
        w, h, x, y = crops[i][0], crops[i][1], crops[i][2], crops[i][3]
        cropped_imgs[i] = imgs_tensor[i][:, y:y+h, x:x+w]
    
    return cropped_imgs

def get_masks(pha):
    """
        pha: V*1*H*W
    """
    pha = pha.permute((0,2,3,1))
    pha_np = pha.to(torch.device('cpu')).numpy()
    masks = []

    for i in range(5):
        mask = np.uint8(pha_np[i]*255)
        masks.append(mask)

    return masks

def get_depth_init_tensor(preds, keyword="coarse_depth_map"):
    """
        return : B*V*H*W
    """
    depth = []
    for key in preds.keys():
        if keyword not in key:
            continue
        tmp = preds[key]
        tmp = F.interpolate(preds[key], size=None, scale_factor=(4,4))
        depth.append(tmp[0,0])

    depth_tensor = torch.stack(depth, 0).unsqueeze(0)

    return depth_tensor


def depth_tensor2numpy(depths_tensor, add_rsize=1):
    depths = []
    view_num = depths_tensor.shape[1]
    depths_tensor = depths_tensor.to(torch.device('cpu'))
    # depths_tensor = F.interpolate(depths_tensor, size=None, scale_factor=(add_rsize, add_rsize))
    for i in range(view_num):
        tmp = depths_tensor[0][i].numpy()
        # tmp = 1.0 / tmp
        # maxn = np.max(tmp)#1.6###1.6
        # minn = np.min(tmp)#0.7###1.0
        # maxn = 1.6
        # minn = 0.6
        # print('max:{},min:{}'.format(maxn, minn))
        # print(f"delta: {maxn-minn}")
        # tmp = (tmp - minn) / (maxn - minn) * 255.0
        # tmp = tmp.astype('uint8')
        # tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_RAINBOW)
        
        # tmp_ori = tmp_ori * (alpha[i][:,:,0]/255)
        
        depths.append(tmp)

    
    return depths


def get_crops_from_mask(mask_tensor, base_size):
    num, dim, height, width = mask_tensor.shape
    maxh = 0
    maxw = 0
    rects = []
    crops = []
    
    for i in range(num):
        mask = mask_tensor[i, 0]
        # print(mask.shape)
        idx = torch.nonzero(mask)
        # print(idx.shape)
        min_y, min_x = torch.min(idx[ :, 0]), torch.min(idx[ :, 1])
        max_y, max_x = torch.max(idx[ :, 0]), torch.max(idx[ :, 1])
        max_x, max_y = int(min(width, max_x)), int(min(height, max_y))
        min_x, min_y = int(max(min_x, 0)), int(max(min_y, 0))
        w = max_x - min_x
        h = max_y - min_y
        if w > maxw: maxw = w
        if h > maxh: maxh = h
        # print((w, h, min_x, min_y))
        rects.append((w, h, min_x, min_y))
    
    maxw = int(int((maxw + base_size - 1)/base_size) * base_size)
    maxh = int(int((maxh + base_size - 1)/base_size) * base_size)
    if maxw > width: maxw = maxw - base_size
    if maxh > height: maxh = maxh - base_size

    for i in range(num):
        w, h, x, y = rects[i]
        newx, newy = x+w-maxw, y+h-maxh
        if newx < 0: newx = 0
        if newy < 0: newy = 0
        # print((maxw, maxh, newx, newy))
        crops.append((maxw, maxh, newx, newy))

    return crops

class DepthEstimation_forRGBD():
    # 要求输入img为RGB

    def __init__(self, view_num, backgrounds, cams, matting_model_path, fmn_model_path, device):
        # paras to read in
        # initialize backgrounds
        self.num_view = view_num
        self.device = device
        self.bgrs = backgrounds.copy()
        self.bgrs = np.stack(self.bgrs)
        self.bgrs_tensor = torch.tensor(self.bgrs).permute(0,3,1,2).float().to(device)
        self.bgrs_tensor_m = self.bgrs_tensor/255.0     # V*3*H*W
        self.cams = cams.copy()

        # Crop: no model
        
        # Matting: load model
        self.Matting_model = torch.jit.load(matting_model_path).eval().to(device)

        # build model
        self.FastMVSNet_model = FastMVSNet_singleframe()
        self.FastMVSNet_model = nn.DataParallel(self.FastMVSNet_model).to(self.device)
        # self.FastMVSNet_model = self.FastMVSNet_model.module
        # # self.FastMVSNet_model = self.FastMVSNet_model.eval().to(device)
        stat_dict = torch.load(fmn_model_path, map_location=torch.device("cpu"))
        self.FastMVSNet_model.load_state_dict(stat_dict.pop("model"), strict = False)


    def getRGBD(self, imgdata, crop=True, checkConsistancy=False):
        """
            ref crop + matting + depth(resize=0.5)
        """
        times = 1   # for velocity test of model
        r_scale = 0.5

        ori_imgs = imgdata.imgs.copy()
        cams = [self.cams[i].copy() for i in range(self.num_view)]

        # ref_crop = [(720, 880, 850, 189), (720, 880, 780, 111), (720, 880, 680, 198), (720, 880, 550, 200), (720, 880, 491, 159)]
        # ref_crop = [(768, 896, 850, 170), (768, 896, 780, 111), (768, 896, 680, 170), (768, 896, 550, 180), (768, 896, 491, 159)]
        ref_crop = [(768, 896, 768, 173), (768, 896, 695, 95), (768, 896, 613, 182), (768, 896, 567, 184), (768, 896, 443, 143)]
        ref_crop_tensor = [torch.tensor(crop).to(self.device) for crop in ref_crop]
        ref_crop_tensor=torch.stack(ref_crop_tensor, 0)
        

        # Crop to get crops(w,h,x,y), cropped_imgs
        # crop_st = time.time()
        if crop:
            # self.crops, maxw, maxh = get_crops(self.imgs, self.bgrs, 64/r_scale)
            crops = ref_crop
            cropped_imgs = crop_images(ori_imgs, crops)
            # Q: self.bgrs_tensor_m:cuda, crops:cpu->cropped_bgrs:cpu 
            # => copy cropped_bgrs every time
            cropped_bgrs = crop_bgrs_tensor(self.bgrs_tensor_m, ref_crop_tensor, self.device)
            cropped_cams = adjust_cam_para(cams, crops)
        else:
            crops = []
            cropped_imgs = [ori_imgs[i].copy() for i in range(self.num_view)]
            cropped_bgrs = self.bgrs
            cropped_cams = cams
        
        # crop_et = time.time()

        # resize cams paras to r_scale for FastMVSNet/Matting
        cropped_cams = [scale_camera(cropped_cams[i], scale=r_scale) for i in range(5)]

        # to tensor to device: cropped_bgrs cropped_imgs
        # matting tensor
        imgs = np.stack(cropped_imgs, axis=0)
        imgs_tensor = torch.tensor(imgs).permute(0,3,1,2).float().to(self.device)
        imgs_tensor_m = (imgs_tensor/255.0)
        
        # bgrs_tensor_m = cropped_bgrs.to(self.device)
        bgrs_tensor_m = cropped_bgrs


        if crop:
            cam_params_list = np.stack(cropped_cams, axis=0)
            cam_params_list = torch.tensor(cam_params_list).float().to(self.device)
            cams_tensor = cam_params_list.unsqueeze(0)  #B*V*2*4*4
            imgs_tensor_d = F.interpolate(imgs_tensor_m, size=None, scale_factor=(r_scale, r_scale))
            imgs_tensor_d = imgs_tensor_d.unsqueeze(0)  # B*V*H*W
            
        else:
            cropped_imgs= [cv2.resize(cropped_imgs[i], None, fx=r_scale, fy=r_scale) for i in range(5)]
            cropped_images, cropped_cams = crop_dtu_input(cropped_imgs, cropped_cams,height=imgs[0].shape[0]*r_scale, width=imgs[0].shape[1]*r_scale, base_image_size=64, depth_image=None)
            img_list = np.stack(cropped_images, axis=0)
            cam_params_list = np.stack(cropped_cams, axis=0)
            img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
            cam_params_list = torch.tensor(cam_params_list).float()

            imgs_tensor_d = img_list.unsqueeze(0).to(self.device)
            cams_tensor = cam_params_list.unsqueeze(0).to(self.device)  # B*V*H*W

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

        # print(alpha_masks.shape) # torch.Size([5, 1, 1080, 1920])

        # run model of depths and masks
        with torch.no_grad():
            # for i in range(3):
            #     preds = self.FastMVSNet_model(imgs_tensor = imgs_tensor_d, cams_tensor = cams_tensor, img_scales=(0.25, 0.5, 1.0), inter_scales=(0.75, 0.15, 0.15), blending=None, isGN=False, isTest=True)

            # torch.cuda.synchronize()
            # depth_st = time.time()
            for i in range(times):
                preds = self.FastMVSNet_model(imgs_tensor = imgs_tensor_d, cams_tensor = cams_tensor, img_scales=(0.25, 0.5, 1.0), inter_scales=(0.75, 0.15, 0.15), blending=None, isGN=False, isTest=True)
            # torch.cuda.synchronize()
            # depth_et = time.time()
        
        # depth_dict_st = time.time()
        depth_tensor = get_depth_init_tensor(preds, "coarse_depth_map") # B*V*H*W
        # get_depth_et = time.time()

        # mask: tensor->numpy
        masks = get_masks(alpha_masks)
        # mask_t2n_et = time.time()

        # Depth Consistancy
        if checkConsistancy:
            # consist_st = time.time()
            _, _, imgH, imgW = depth_tensor.shape
            # print(cropped_cams[i][1].shape, cropped_cams[i][0].shape)
            consist_cams = [MyPerspectiveCamera(cropped_cams[i][1][:3, :3], cropped_cams[i][0], imgH, imgW, device = self.device) for i in range(self.num_view)]
            refined_depth_tensor, refined_mask_tensor = ConsistancyChecker.getMeanCorrectDepth(
            [consist_cams], depth_tensor, pix_thre=1.0, dep_thre=0.01, view_thre=2, 
            absoluteDepth=False)
            refined_masked_depth_tensor = refined_depth_tensor*refined_mask_tensor
            # consist_et = time.time()
            
            # depth_t2n_st = time.time()
            depths = depth_tensor2numpy(refined_masked_depth_tensor, add_rsize=1/r_scale)
            # depth_t2n_et = time.time()
        
        else:
            # depth_t2n_st = time.time()
            depths = depth_tensor2numpy(depth_tensor, add_rsize=1/r_scale)
            # depth_t2n_et = time.time()

        depths = [cv2.resize(depths[i], None, fx=1/r_scale, fy=1/r_scale) for i in range(5)]

        et = time.time()

        # print(f"crop: {crop_et - crop_st}")
        # print(f"resize and trans: {trans_et - crop_et}")
        # print(f"matting: {(mattint_et - matting_st)/times}")
        # print(f"depth_estimation: {(depth_et - depth_st)/times}")
        # print(f"select depth: {get_depth_et - depth_dict_st}")
        # print(f"trans mask: {mask_t2n_et - get_depth_et}")
        # if checkConsistancy: print(f"check consistacy: {consist_et - consist_st}")
        # print(f"trans depth: {depth_t2n_et - depth_t2n_st}")
        # print(f"resize depth: {et - depth_t2n_et}")


        return {
            "num_view": self.num_view,
            "imgs": ori_imgs,
            "depths": depths, 
            "masks": masks, 
            "crops": crops
        }

    def getRGBD_crop_after_matting(self, imgdata, crop=True, checkConsistancy=False):
        times = 1
        r_scale = 0.5

        # st = time.time()

        # ori_imgs = imgdata.copy()
        ori_imgs = imgdata.imgs.copy()
        cams = [self.cams[i].copy() for i in range(self.num_view)]

        # trans for matting

        # st = time.time()
        ori_imgs = np.stack(ori_imgs, axis=0)
        # st = time.time()
        imgs_tensor = torch.tensor(ori_imgs).permute(0,3,1,2).float().to(self.device)   # torch.Size([5, 3, 1080, 1920])
        imgs_tensor_m = imgs_tensor/255


        # trans_m_et = time.time()

        # matting
        # for i in range(3):
        #     alpha_masks = self.Matting_model(imgs_tensor_m, bgrs_tensor_m)[0]
        
        # torch.cuda.synchronize()
        # matting_st = time.time()
        for i in range(times):
            alpha_masks = self.Matting_model(imgs_tensor_m, self.bgrs_tensor_m)[0]
        # torch.cuda.synchronize()
        # mattint_et = time.time()

        # change bgr before depth estimation
        # green = torch.tensor([0, 255, 0]).view(1, 3, 1, 1).to(self.device)
        # imgs_tensor = alpha_masks * imgs_tensor + (1 - alpha_masks) * green
        
        # get crops by mask
        crops = get_crops_from_mask(alpha_masks, 64/r_scale)

        # get_crop_et = time.time()

        # crop imgs and cams
        crop_w, crop_h = crops[0][0], crops[0][1]
        imgs_tensor_d = torch.zeros((5, 3, crop_h, crop_w))
        for i in range(self.num_view):
            x, y = crops[i][2], crops[i][3]
            imgs_tensor_d[i,:] = imgs_tensor[i, :, y:y+crop_h, x:x+crop_w]
        imgs_tensor_d = F.interpolate(imgs_tensor_d, (int(crop_h*r_scale), int(crop_w*r_scale)), mode="bilinear")
        imgs_tensor_d = imgs_tensor_d.unsqueeze(0)

        cropped_cams = adjust_cam_para(cams, crops)
        cropped_cams = [scale_camera(cropped_cams[i], scale=r_scale) for i in range(5)]
        cam_params = np.stack(cropped_cams, axis=0)
        cam_params = torch.tensor(cam_params).float().to(self.device)
        cams_tensor = cam_params.unsqueeze(0)

        # trans_d_et = time.time()

        # run model of depths and masks
        with torch.no_grad():
            # for i in range(3):
            #     preds = self.FastMVSNet_model(imgs_tensor = imgs_tensor_d, cams_tensor = cams_tensor, img_scales=(0.25, 0.5, 1.0), inter_scales=(0.75, 0.15, 0.15), blending=None, isGN=False, isTest=True)

            # torch.cuda.synchronize()
            # depth_st = time.time()
            for i in range(times):
                # preds: dict
                preds = self.FastMVSNet_model(imgs_tensor = imgs_tensor_d, cams_tensor = cams_tensor, img_scales=(0.25, 0.5, 1.0), inter_scales=(0.75, 0.15, 0.15), blending=None, isGN=False, isTest=True)
            # torch.cuda.synchronize()
            # depth_et = time.time()
        
        # depths/masks: tensor->numpy
        # for key in preds:
        #     if "coarse_depth_map" in key:
        #         preds[key] = F.interpolate(preds[key], (crop_h, crop_w), mode="bilinear")
        masks = get_masks(alpha_masks)
        depth_tensor = get_depth_init_tensor(preds, "coarse_depth_map") # B*V*H*W

        # Depth Consistancy
        if checkConsistancy:
            # consist_st = time.time()
            _, _, imgH, imgW = depth_tensor.shape
            # print(cropped_cams[i][1].shape, cropped_cams[i][0].shape)
            consist_cams = [MyPerspectiveCamera(cropped_cams[i][1][:3, :3], cropped_cams[i][0], imgH, imgW, device = self.device) for i in range(self.num_view)]
            refined_depth_tensor, refined_mask_tensor = ConsistancyChecker.getMeanCorrectDepth(
            [consist_cams], depth_tensor, pix_thre=1.0, dep_thre=0.01, view_thre=2, 
            absoluteDepth=False)
            refined_masked_depth_tensor = refined_depth_tensor*refined_mask_tensor
            # consist_et = time.time()
            
            # depth_t2n_st = time.time()
            depths = depth_tensor2numpy(refined_masked_depth_tensor, add_rsize=1/r_scale)
            # depth_t2n_et = time.time()
        
        else:
            # depth_t2n_st = time.time()
            depths = depth_tensor2numpy(depth_tensor, add_rsize=1/r_scale)
            # depth_t2n_et = time.time()

        depths = [cv2.resize(depths[i], None, fx=1/r_scale, fy=1/r_scale) for i in range(5)]

        # get_back_et = time.time()

        # et = time.time()

        # print(f"trans matting(1080p*2) time: {trans_m_et - st}s")
        # print(f"matting time: {(mattint_et - matting_st)/times}s")
        # print(f"get crop from mask: {get_crop_et- mattint_et}")
        # print(f"crop and trans: {trans_d_et - get_crop_et}")
        # print(f"depth_estimation time: {(depth_et - depth_st)/times}s")
        # print(f"trans: {get_back_et - depth_et}s")
        # print(f"resize depth: {et - get_back_et}s")


        return {
            "num_view": self.num_view,
            "imgs": ori_imgs,
            "depths": depths, 
            "masks": masks, 
            "crops": crops
        }
            
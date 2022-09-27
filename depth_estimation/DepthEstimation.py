from email.mime import base
import os
import sys
from turtle import width
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

sys.path.insert(1, os.path.dirname(__file__)+'/.')

from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender

from fastmvsnet.model_sf import FastMVSNet_singleframe
from data_preparation import adjust_cam_para, scale_camera
from fastmvsnet.utils.preprocess import  crop_dtu_input
from tools.propagation import warping_propagation_singleframe
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
        # newx, newy = x+w-maxw, y+h-maxh
        newx, newy = x - (maxw - w)/2, y - (maxh - h)/2 
        if newx < 0: newx = 0
        if newy < 0: newy = 0
        if newx + maxw > img_w: newx = img_w - maxw
        if newy + maxh > img_h: newy = img_h - maxh

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

def crop_imgs_tensor(imgs_tensor, crops, device):
    """
    imgs_tensor: (tensor) V*3*H*W
    return: (tensor) V*3*crop_h*crop_w
    """
    num, C, H, W = imgs_tensor.shape
    width, height = crops[0][0], crops[0][1]
    cropped_imgs = torch.ones((num, C, height, width)).to(device)
    
    for i in range(num):
        w, h, x, y = crops[i][0], crops[i][1], crops[i][2], crops[i][3]
        cropped_imgs[i] = imgs_tensor[i][:, y:y+h, x:x+w]
    
    return cropped_imgs

def get_masks(pha):
    """
        pha: V*1*H*W
        return: (list)V*H*W*1
    """
    pha = pha.permute((0,2,3,1))    # V*H*W*1
    pha_np = pha.to(torch.device('cpu')).numpy()
    masks = []

    for i in range(pha.shape[0]):
        mask = np.uint8(pha_np[i]*255)
        masks.append(mask)

    return masks

def get_depth_init_tensor(preds, keyword="coarse_depth_map"):
    """
        input: dict
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
    """
        Trans depths from tensor to numpy
        and change all d = 0 to d[d>0].min() (for next step's resize)
        
        input: depths_tensor
        return: (list)numpy
    """
    depths = []
    view_num = depths_tensor.shape[1]
    depths_tensor = depths_tensor.to(torch.device('cpu'))
    # depths_tensor = F.interpolate(depths_tensor, size=None, scale_factor=(add_rsize, add_rsize))
    for i in range(view_num):
        tmp = depths_tensor[0][i].detach().numpy()
        # tmp[tmp==0] = tmp[tmp>0].max()

        # tmp = 1.0 / tmp
        # tmp[tmp < 0] = 0.5
        # maxn = np.max(tmp)#1.6###1.6
        # minn = np.min(tmp)#0.7###1.0
        # maxn = 1.6
        # minn = 0.5
        # print('max:{},min:{}'.format(maxn, minn))
        # print(f"delta: {maxn-minn}")
        # tmp = (tmp - minn) / (maxn - minn) *255.0
        # tmp = tmp*255.0
        # tmp = tmp.astype('uint8')
        # tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_RAINBOW)
        
        # tmp_ori = tmp_ori * (alpha[i][:,:,0]/255)

        depths.append(tmp)

    
    return depths


def get_crops_from_mask(mask_tensor, base_size):
    """
        input:  mask_tensor  masks got from matting
        return: (list) (maxw, maxh, newx, newy)
    """
    num, dim, height, width = mask_tensor.shape
    maxh = 0
    maxw = 0
    rects = []
    crops = []
    
    for i in range(num):
        mask = mask_tensor[i, 0]    # H, W
        idx = torch.nonzero(mask)

        if idx.shape[0] == 0:
            maxw, maxh = width, height
            for i in range(num):
                rects.append([width, height, 0, 0])
            break

        min_y, min_x = torch.min(idx[ :, 0]), torch.min(idx[ :, 1])
        max_y, max_x = torch.max(idx[ :, 0]), torch.max(idx[ :, 1])
        max_x, max_y = min(width, max_x), min(height, max_y)
        min_x, min_y = max(min_x, 0), max(min_y, 0)
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
        newx, newy = x + (w - maxw)/2, y + (h - maxh)/2
        if newx < 0: newx = 0
        if newy < 0: newy = 0
        if newx + maxw > width: newx = width - maxw
        if newy + maxh > height: newy = height - maxh
        # print((maxw, maxh, newx, newy))
        crops.append((int(maxw), int(maxh), int(newx), int(newy)))

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
        self.bgrs_tensor_01 = self.bgrs_tensor/255.0     # V*3*H*W
        self.cams = cams.copy() # intrs：cams[idx][1][:3, :3]; extrs: cams[idx][0] 

        self.ref_crops = [(768, 896, 768, 173), (768, 896, 695, 95), (768, 896, 613, 182), (768, 896, 567, 184), (768, 896, 443, 143)]


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


    def getRGBD(self, imgdata, crop=True, isGN=False, checkConsistancy=False, propagation=False, checkSecondConsistancy=False):
        """
            input: ImageData
            return: (dict)
                "num_view": num_view    int
                "imgs":     src_imgs    H * W * 3 (uint8)
                "depths":   depths      crop_H * crop_W (float32)
                "masks":    masks       crop_H * crop_W * 1 (uint8)
                "crops":    crops       tuple   (crop_W, crop_H, start_x, start_y)
        """
        times = 1   # for velocity test of model
        r_scale = 0.5

        src_imgs = imgdata.imgs.copy()
        cams = [self.cams[i].copy() for i in range(self.num_view)]

        # ref_crop = [(720, 880, 850, 189), (720, 880, 780, 111), (720, 880, 680, 198), (720, 880, 550, 200), (720, 880, 491, 159)]
        ref_crop = [(768, 896, 768, 173), (768, 896, 695, 95), (768, 896, 613, 182), (768, 896, 567, 184), (768, 896, 443, 143)]
        ref_crop_tensor = [torch.tensor(crop).to(self.device) for crop in ref_crop]
        ref_crop_tensor = torch.stack(ref_crop_tensor, 0)   # 5*(crop_w, crop_h, x, y)
        

        # Crop to get crops(w,h,x,y)
        # cropped_imgs: V*3*crop_h*crop_w
        # cropped_bgrs: V*3*crop_h*crop_w

        # Get crop parameters and crop imgs/bgrs/cams
        # crop_st = time.time()
        if crop:
            # crops, maxw, maxh = get_crops(self.imgs, self.bgrs, 64/r_scale)  # calculate crops
            crops = ref_crop
            cropped_imgs = crop_images(src_imgs, crops)
            # Q: self.bgrs_tensor_m:cuda, crops:cpu->cropped_bgrs:cpu 
            # => copy cropped_bgrs every time
            cropped_bgrs = crop_imgs_tensor(self.bgrs_tensor_01, ref_crop_tensor, self.device)
            cropped_cams = adjust_cam_para(cams, crops)
        else:
            crops = []
            cropped_imgs = [src_imgs[i].copy() for i in range(self.num_view)]
            cropped_bgrs = self.bgrs
            cropped_cams = cams
        
        # crop_et = time.time()

        # Resize cams paras to r_scale for FastMVSNet/Matting
        s_cropped_cams = [scale_camera(cropped_cams[i], scale=r_scale) for i in range(5)]
        # print(s_cropped_cams[0], "\n", cropped_cams[0])

        # Imgs to tensor to device: cropped_bgrs cropped_imgs
        # matting tensor
        imgs = np.stack(cropped_imgs, axis=0)
        imgs_tensor = torch.tensor(imgs).permute(0,3,1,2).float().to(self.device)   # V*3*H*W
        imgs_tensor_m = (imgs_tensor/255.0)
        
        # If backgrounds keep unchanged, get from class directly(no need to put to device)
        # bgrs_tensor_m = cropped_bgrs.to(self.device)
        bgrs_tensor_m = cropped_bgrs


        if crop:
            s_cam_params_list = np.stack(s_cropped_cams, axis=0)    # 5*2*4*4
            s_cam_params_list = torch.tensor(s_cam_params_list).float().to(self.device)
            s_cams_tensor = s_cam_params_list.unsqueeze(0)  # B*V*2*4*4
            imgs_tensor_d = F.interpolate(imgs_tensor_m, size=None, scale_factor=(r_scale, r_scale))
            imgs_tensor_d = imgs_tensor_d.unsqueeze(0)  # B*V*3*H*W

            # get original cam paras
            cams_tensor = s_cams_tensor.clone()
            cams_tensor[:, :, 1, 0, 0] = cams_tensor[:, :, 1, 0, 0] / r_scale
            cams_tensor[:, :, 1, 1, 1] = cams_tensor[:, :, 1, 1, 1] / r_scale
            cams_tensor[:, :, 1, 0, 2] = cams_tensor[:, :, 1, 0, 2] / r_scale
            cams_tensor[:, :, 1, 1, 2] = cams_tensor[:, :, 1, 1, 2] / r_scale
            
        else:
            # 未经过crop需要裁边
            cropped_imgs= [cv2.resize(cropped_imgs[i], None, fx=r_scale, fy=r_scale) for i in range(5)]
            cropped_images, cropped_cams = crop_dtu_input(cropped_imgs, cropped_cams,height=imgs[0].shape[0]*r_scale, width=imgs[0].shape[1]*r_scale, base_image_size=64, depth_image=None)
            img_list = np.stack(cropped_images, axis=0)
            s_cam_params_list = np.stack(cropped_cams, axis=0)
            img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
            s_cam_params_list = torch.tensor(s_cam_params_list).float()

            imgs_tensor_d = img_list.unsqueeze(0).to(self.device)       # B*V*3*H*W
            s_cams_tensor = s_cam_params_list.unsqueeze(0).to(self.device)  # B*V*2*4*4

        # trans_et = time.time()
        
        # Run model of depths and masks
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

        with torch.no_grad():
            # for i in range(3):
            #     preds = self.FastMVSNet_model(imgs_tensor = imgs_tensor_d, cams_tensor = cams_tensor, img_scales=(0.25, 0.5, 1.0), inter_scales=(0.75, 0.15, 0.15), blending=None, isGN=False, isTest=True)

            # torch.cuda.synchronize()
            # depth_st = time.time()
            for i in range(times):
                # preds: dict
                preds = self.FastMVSNet_model(imgs_tensor = imgs_tensor_d, cams_tensor = s_cams_tensor, img_scales=(0.25, 0.5, 1.0), inter_scales=(0.75, 0.15, 0.15), blending=None, isGN=isGN, isTest=True)
            # torch.cuda.synchronize()
            # depth_et = time.time()
        
        # depth_dict_st = time.time()
        depth_tensor = get_depth_init_tensor(preds, "coarse_depth_map") # dict -> B*V*H*W
        # get_depth_et = time.time()


        # Depth Consistancy
        if checkConsistancy:
            # consist_st = time.time()
            
            _, _, imgH, imgW = depth_tensor.shape   # B*V*H*W
            # print(f"imgH:{imgH}, imgW: {imgW}")
            # print(cropped_cams[i][1].shape, cropped_cams[i][0].shape)
            consist_cams = [MyPerspectiveCamera(s_cropped_cams[i][1][:3, :3], s_cropped_cams[i][0], imgH, imgW, device = self.device) for i in range(self.num_view)]

            refined_depth_tensor, refined_mask_tensor = ConsistancyChecker.getMeanCorrectDepth(
            [consist_cams], depth_tensor, pix_thre=1.0, dep_thre=0.01, view_thre=3, 
            absoluteDepth=False)    
            # refined_depth_tensor: B*V*H*W; refined_mask_tensor: B*V*H*W
            refined_masked_depth_tensor = refined_depth_tensor*refined_mask_tensor

            # print("checkConsistancy1:")
            # for k in range(5):
            #     tmp = refined_masked_depth_tensor.clone()
            #     tmp[tmp==0] = 100
            #     print(tmp[:, k, ...].max(), tmp[:, k, ...].min())
            
            # consist_et = time.time()
            
            if propagation:
                # propagation_st = time.time()
                # mask
                # alpha_masks_4prop = F.interpolate(alpha_masks, size=None, scale_factor=(r_scale, r_scale))
                # refined_masked_depth_tensor = refined_masked_depth_tensor*(alpha_masks_4prop.permute(1,0,2,3)[0].round())
                
                refined_masked_depth_tensor = warping_propagation_singleframe(imgs_tensor_d, refined_masked_depth_tensor, refined_mask_tensor,  device=self.device)

                # print("propagation:")
                # for k in range(5):
                #     tmp = refined_masked_depth_tensor.clone()
                #     tmp[tmp==0] = 100
                #     print(tmp[:, k, ...].max(), tmp[:, k, ...].min())

                # propagation_et = time.time()

            # check 2nd Consistancy to find fault(block)
            if checkSecondConsistancy:
                # consist_2_st = time.time()
                refined_depth_tensor, refined_mask_tensor = ConsistancyChecker.getMeanCorrectDepth(
                [consist_cams], refined_masked_depth_tensor, pix_thre=2.0, dep_thre=0.02, view_thre=2, 
                absoluteDepth=False)    
                # refined_depth_tensor: B*V*H*W; refined_mask_tensor: B*V*H*W
                refined_masked_depth_tensor = refined_depth_tensor*refined_mask_tensor

                # print("check2Consistancy:")
                # for k in range(5):
                #     tmp = refined_masked_depth_tensor.clone()
                #     tmp[tmp==0] = 100
                #     print(tmp[:, k, ...].max(), tmp[:, k, ...].min())

                # consist_2_et = time.time()
        else:
            refined_masked_depth_tensor = depth_tensor

        refined_mask_tensor = F.interpolate(refined_mask_tensor.float(), size=None, scale_factor=(1/r_scale, 1/r_scale), mode='bilinear')
        # print(refined_mask_tensor.shape, refined_mask_tensor.dtype) # B * V * src_H * src_W  float32

        refined_mask_tensor[refined_mask_tensor < 1] = 0
        alpha_masks_tensor = alpha_masks.permute((0,2,3,1)).squeeze()  # V*H*W*1 -> V*H*W
        out_masks_tensor = alpha_masks_tensor * refined_mask_tensor[0]
        out_mask_np = out_masks_tensor.to(torch.device('cpu')).numpy()
        masks = [np.uint8(out_mask_np[i]*255) for i in range(self.num_view)]

        # mask_t2n_et = time.time()
        
        # depth_t2n_st = time.time()
        depths = depth_tensor2numpy(refined_masked_depth_tensor, add_rsize=1/r_scale)
        # depth_t2n_et = time.time()

        # Upsample to original size
        depths = [cv2.resize(depths[i], None, fx=1/r_scale, fy=1/r_scale) for i in range(5)]

        # et = time.time()

        # print(f"crop: {crop_et - crop_st}")
        # print(f"resize and trans to device: {trans_et - crop_et}")
        # print(f"matting: {(mattint_et - matting_st)/times}")
        # print(f"depth_estimation: {(depth_et - depth_st)/times}")
        # print(f"select depth from dict and turn to tensor: {get_depth_et - depth_dict_st}")
        # print(f"mask t2n: {mask_t2n_et - get_depth_et}")
        # if checkConsistancy: print(f"check consistacy: {consist_et - consist_st}")
        # if propagation: print(f"propagation: {propagation_et - propagation_st}")
        # if checkSecondConsistancy:
        #     print(f"check 2nd consistancy: {consist_2_et - consist_2_st}")
        # print(f"depth t2n: {depth_t2n_et - depth_t2n_st}")
        # print(f"resize depth: {et - depth_t2n_et}")


        return {
            "num_view": self.num_view,
            "imgs": src_imgs,
            "depths": depths, 
            "masks": masks, 
            "crops": crops
        }
            

    def getRGBD_test(self, imgdata, crop=True, isGN=False, checkConsistancy=False, propagation=False, checkSecondConsistancy=False):
        """
            input: ImageData
            return: (dict)
                "num_view": num_view    int
                "imgs":     src_imgs    crop_H * crop_W * 3 (uint8)
                "depths":   depths      crop_H * crop_W (float32)
                "masks":    masks       crop_H * crop_W * 1 (uint8)
                "crops":    crops       tuple   (crop_W, crop_H, start_x, start_y)
        """
        times = 1   # for velocity test of model
        r_scale = 0.5

        src_imgs = imgdata.imgs.copy()
        cams = [self.cams[i].copy() for i in range(self.num_view)]        

        if crop:
            # Build new crop tuples
            imgs_np = np.stack(src_imgs, axis=0)
            imgs_tensor = torch.tensor(imgs_np).permute(0,3,1,2).float().to(self.device)
            imgs_tensor_01 = imgs_tensor/255.0

            # Matting
            alpha_masks = self.Matting_model(imgs_tensor_01, self.bgrs_tensor_01)[0]

            # Get crops from matting masks
            crops = get_crops_from_mask(alpha_masks, 64/r_scale)

            self.ref_crops = crops

            cropped_alpha_masks_tensor = crop_imgs_tensor(alpha_masks, crops, self.device)  # V*1*crop_H*crop_W
            cropped_alpha_masks_tensor = cropped_alpha_masks_tensor.permute((0,2,3,1)).squeeze()  # V*crop_H*crop_W*1 -> V*crop_H*crop_W

            cropped_imgs = crop_images(src_imgs, crops)

            crop_w, crop_h = crops[0][0], crops[0][1]
            cropped_imgs_tensor = crop_imgs_tensor(imgs_tensor_01, crops, self.device)

            s_cropped_imgs_tensor = F.interpolate(cropped_imgs_tensor, (int(crop_h*r_scale), int(crop_w*r_scale)), mode="bilinear")
            s_cropped_imgs_tensor = s_cropped_imgs_tensor.unsqueeze(0)

            cropped_cams = adjust_cam_para(cams, crops)
            s_cropped_cams = [scale_camera(cropped_cams[i], scale=r_scale) for i in range(5)]
            s_cropped_cams_np = np.stack(s_cropped_cams, axis=0)
            s_cropped_cams_tensor = torch.tensor(s_cropped_cams_np).float().to(self.device)
            s_cropped_cams_tensor = s_cropped_cams_tensor.unsqueeze(0)
        
        else:
            crops = self.ref_crops
            crop_tensor = [torch.tensor(crop).to(self.device) for crop in crops]
            crop_tensor = torch.stack(crop_tensor, 0)   # 5*(crop_w, crop_h, x, y)
            
            cropped_bgrs_tensor = crop_imgs_tensor(self.bgrs_tensor_01, crop_tensor, self.device)
            cropped_imgs = crop_images(src_imgs, crops)
            cropped_cams = adjust_cam_para(cams, crops)

            s_cropped_cams = [scale_camera(cropped_cams[i], scale=r_scale) for i in range(5)]

            cropped_imgs_np = np.stack(cropped_imgs, axis=0)
            cropped_imgs_tensor = torch.tensor(cropped_imgs_np).permute(0,3,1,2).float().to(self.device)   # V*3*H*W
            cropped_imgs_tensor = (cropped_imgs_tensor/255.0)

            s_cropped_cams_np = np.stack(s_cropped_cams, axis=0)
            s_cropped_cams_tensor = torch.tensor(s_cropped_cams_np).float().to(self.device)
            s_cropped_cams_tensor = s_cropped_cams_tensor.unsqueeze(0)

            s_cropped_imgs_tensor = F.interpolate(cropped_imgs_tensor, size=None, scale_factor=(r_scale, r_scale))
            s_cropped_imgs_tensor = s_cropped_imgs_tensor.unsqueeze(0)

            cropped_alpha_masks = self.Matting_model(cropped_imgs_tensor, cropped_bgrs_tensor)[0]
            cropped_alpha_masks_tensor = cropped_alpha_masks.permute(0,2,3,1).squeeze()  # V*H*W*1 -> V*H*W



        preds = self.FastMVSNet_model(imgs_tensor = s_cropped_imgs_tensor, cams_tensor = s_cropped_cams_tensor, img_scales=(0.25, 0.5, 1.0), inter_scales=(0.75, 0.15, 0.15), blending=None, isGN=isGN, isTest=True)

        depth_tensor = get_depth_init_tensor(preds, "coarse_depth_map")

        # Depth Consistancy
        if checkConsistancy:
            # consist_st = time.time()
            
            _, _, imgH, imgW = depth_tensor.shape   # B*V*H*W

            consist_cams = [MyPerspectiveCamera(s_cropped_cams[i][1][:3, :3], s_cropped_cams[i][0], imgH, imgW, device = self.device) for i in range(self.num_view)]
            refined_depth_tensor, refined_mask_tensor = ConsistancyChecker.getMeanCorrectDepth(
            [consist_cams], depth_tensor, pix_thre=1.0, dep_thre=0.01, view_thre=3, 
            absoluteDepth=False)    
            # refined_depth_tensor: B*V*H*W; refined_mask_tensor: B*V*H*W
            refined_masked_depth_tensor = refined_depth_tensor*refined_mask_tensor
            
            # consist_et = time.time()
            
            if propagation:
                # propagation_st = time.time()
                # mask

                depths = warping_propagation_singleframe(s_cropped_imgs_tensor, refined_masked_depth_tensor, refined_mask_tensor, device=self.device)
                refined_masked_depth_tensor = depths

                # propagation_et = time.time()

                # check 2nd Consistancy to find fault(block)
                if checkSecondConsistancy:
                    # consist_2_st = time.time()

                    refined_depth_tensor, refined_mask_tensor = ConsistancyChecker.getMeanCorrectDepth(
                    [consist_cams], refined_masked_depth_tensor, pix_thre=2.0, dep_thre=0.02, view_thre=2, 
                    absoluteDepth=False)    
                    # refined_depth_tensor: B*V*H*W; refined_mask_tensor: B*V*H*W
                    refined_masked_depth_tensor = refined_depth_tensor*refined_mask_tensor

                    # consist_2_et = time.time()
        else:
            refined_masked_depth_tensor = depth_tensor


        # if checkConsistancy, use refined mask + matting mask
        if checkConsistancy:
            refined_mask_tensor = F.interpolate(refined_mask_tensor.float(), size=None, scale_factor=(1/r_scale, 1/r_scale), mode='bilinear')
            refined_mask_tensor[refined_mask_tensor < 1] = 0
            out_masks_tensor = (cropped_alpha_masks_tensor * refined_mask_tensor[0])
        else:
            out_masks_tensor = cropped_alpha_masks_tensor
            
        out_mask_np = out_masks_tensor.to(torch.device('cpu')).numpy()
        masks = [np.uint8(out_mask_np[i]*255) for i in range(self.num_view)]


        depths = depth_tensor2numpy(refined_masked_depth_tensor, add_rsize=1/r_scale)

        depths = [cv2.resize(depths[i], None, fx=1/r_scale, fy=1/r_scale) for i in range(5)]


        # print(f"crop: {crop_et - crop_st}")
        # print(f"resize and trans to device: {trans_et - crop_et}")
        # print(f"matting: {(mattint_et - matting_st)/times}")
        # print(f"depth_estimation: {(depth_et - depth_st)/times}")
        # print(f"select depth from dict and turn to tensor: {get_depth_et - depth_dict_st}")
        # print(f"mask t2n: {mask_t2n_et - get_depth_et}")
        # if checkConsistancy: print(f"check consistacy: {consist_et - consist_st}")
        # if propagation: print(f"propagation: {propagation_et - propagation_st}")
        # if checkSecondConsistancy:
        #     print(f"check 2nd consistancy: {consist_2_et - consist_2_st}")
        # print(f"depth t2n: {depth_t2n_et - depth_t2n_st}")
        # print(f"resize depth: {et - depth_t2n_et}")


        return {
            "num_view": self.num_view,
            "imgs": cropped_imgs,
            "depths": depths, 
            "masks": masks, 
            "crops": crops
        }
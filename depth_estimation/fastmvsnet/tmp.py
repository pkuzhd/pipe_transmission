import os
import time

import torch
import torch.nn.functional as F
from picutils import PICTimer
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fastmvsnet.utils.io import *

def test_jacobi():
    timer = PICTimer.getTimer()
    timer.startTimer()
    for _ in range(100):
        a = torch.rand([58982400, 1, 2], device='cuda:1')
        b = torch.rand([58982400, 2, 1], device='cuda:1')
        torch.cuda.synchronize()
        timer.showTime('rand')
        # c = a @ b
        b = b.permute(0, 2, 1)
        d = a * b
        d = torch.sum(d, dim=2)
        # x = torch.sum(c, dim=2)
        # x = torch.sum(x - d)
        torch.cuda.synchronize()
        timer.showTime('compute')
    timer.summary()

def test_gird_sampling():
    timer = PICTimer.getTimer()
    timer.startTimer()

    feature_maps = torch.rand([5, 48, 640, 384], dtype=torch.float32, device='cuda:1')
    grid = torch.rand([5, 245760, 1, 2], dtype=torch.float32, device='cuda:1')
    # feature_maps = torch.load('/home/wjk/workspace/PyProject/FastMVSNet/fastmvsnet/feature_maps').to('cuda:0')
    # grid = torch.load('/home/wjk/workspace/PyProject/FastMVSNet/fastmvsnet/grid_uv').to('cuda:0')
    width, height = 384, 640
    torch.cuda.synchronize()
    timer.showTime('rand')

    def get_features(grid_uv):
        pts_feature = F.grid_sample(feature_maps, grid_uv, mode='bilinear')
        pts_feature = pts_feature.squeeze(3)
        pts_feature = pts_feature.view(1, 5, 48, 245760)
        return pts_feature.detach()

    pts_feature = get_features(grid)
    torch.cuda.synchronize()
    timer.showTime('warping')

    grid[..., 0] -= (1. / float(width - 1)) * 2
    pts_feature_l = get_features(grid)
    grid[..., 0] += (1. / float(width - 1)) * 2

    grid[..., 0] += (1. / float(width - 1)) * 2
    pts_feature_r = get_features(grid)
    grid[..., 0] -= (1. / float(width - 1)) * 2

    grid[..., 1] -= (1. / float(height - 1)) * 2
    pts_feature_t = get_features(grid)
    grid[..., 1] += (1. / float(height - 1)) * 2

    grid[..., 1] += (1. / float(height - 1)) * 2
    pts_feature_b = get_features(grid)
    grid[..., 1] -= (1. / float(height - 1)) * 2
    torch.cuda.synchronize()
    timer.showTime('warping lrbt')

    pts_feature_r -= pts_feature_l
    pts_feature_r *= 0.5
    pts_feature_b -= pts_feature_t
    pts_feature_b *= 0.5
    torch.cuda.synchronize()
    timer.showTime('d_uv')
    timer.summary()


def test_grident_table():
    timer = PICTimer.getTimer()
    timer.startTimer()

    feature_maps = torch.load('/home/wjk/workspace/PyProject/FastMVSNet/fastmvsnet/feature_maps').to('cuda:0')
    width, height = 384, 640
    torch.cuda.synchronize()
    timer.showTime('rand')

    feature_maps_l = torch.zeros_like(feature_maps)
    feature_maps_l[:, :, :-1, :] = feature_maps[:, :, 1:, :]
    feature_maps_r = torch.zeros_like(feature_maps)
    feature_maps_r[:, :, 1:, :] = feature_maps[:, :, :-1, :]
    feature_maps_b = torch.zeros_like(feature_maps)
    feature_maps_b[:, :, :, :-1] = feature_maps[:, :, :, 1:]
    feature_maps_t = torch.zeros_like(feature_maps)
    feature_maps_t[:, :, :, 1:] = feature_maps[:, :, :, :-1]
    torch.cuda.synchronize()
    timer.showTime('prepare')

    feature_maps_r -= feature_maps_l
    feature_maps_r *= 0.5
    feature_maps_b -= feature_maps_t
    feature_maps_b *= 0.5
    torch.cuda.synchronize()
    timer.showTime('d_uv')
    timer.summary()

#test_gird_sampling()
#test_grident_table()


def fore_out_of_back():
    fgbg = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=100, detectShadows=False)
    back = cv2.imread('/home/wjk/workspace/PyProject/MVS2D/demo/lab/rgb_b/000000.png')
    back = cv2.pyrDown(back)
    fore = cv2.imread('/home/wjk/workspace/PyProject/MVS2D/demo/lab/rgb/000000.png')
    fore = cv2.pyrDown(fore)
    cv2.imshow("rgb", back)
    cv2.waitKey()
    cv2.imshow("rgb", fore)
    cv2.waitKey()
    fore_max = fore.copy()

    # get the front mask
    begin_t = time.time()
    mask = fgbg.apply(back)
    print('back: ', time.time()-begin_t)
    begin_t = time.time()
    mask = fgbg.apply(fore)
    print('fore: ', time.time()-begin_t)
    cv2.imshow("mask", mask)
    cv2.waitKey()
    begin_t = time.time()
    # eliminate the noise
    line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), (-1, -1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)
    print('noise: ', time.time()-begin_t)
    cv2.imshow("mask", mask)
    cv2.waitKey()

    begin_t = time.time()
    max_area = -1
    max_rect = None
    # find the max area contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('find contours: ', time.time()-begin_t)
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area < 150:
            continue
        rect = cv2.minAreaRect(contours[c])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(fore, [box], 0, (0, 255, 0), 2)
        #cv2.ellipse(fore, rect, (0, 255, 0), 2, 8)
        cv2.circle(fore, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
        if area > max_area:
            max_area = area
            max_rect = rect

    cv2.imshow('fore', fore)
    cv2.waitKey()

    box = cv2.boxPoints(max_rect)
    max_x, min_x = np.max(box[:,0]), np.min(box[:,0])
    max_y, min_y = np.max(box[:,1]), np.min(box[:,1])
    box = np.array([[min_x, max_y],[min_x, min_y],[max_x, min_y],[max_x, max_y]])
    box = np.int0(box)
    cv2.drawContours(fore_max, [box], 0, (0, 255, 0), 2)
    cv2.circle(fore_max, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)

    cv2.imshow('fore_max', fore_max)
    cv2.waitKey()

#fore_out_of_back()

def test_FFT():
    x = torch.rand(2160, 3840, dtype=torch.float32, device='cuda:1')
    fft2 = torch.fft.fft2(x)

    timer = PICTimer.getTimer()
    timer.startTimer()

    img = cv2.imread('/home/wjk/workspace/PyProject/MVS2D/demo/lab/rgb_b/000004.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(img)
    plt.show()

    img = img.astype('float32')
    img = torch.tensor(img).to('cuda:1')
    torch.cuda.synchronize()
    timer.showTime('rand')

    fft_img = torch.fft.fft2(img)
    torch.cuda.synchronize()
    timer.showTime('fft')

    img = torch.fft.ifft2(fft_img)
    torch.cuda.synchronize()
    timer.showTime('ifft')
    timer.summary()

    img.backward()
    # fft_img = torch.log(torch.abs(fft_img)).cpu().numpy()
    # plt.imshow(fft_img)
    # plt.show()
    #
    # plt.imshow(img.real.cpu().numpy())
    # plt.show()

#test_FFT()

#### test time taken to move data from cpu to cuda
def test_data_copy():
    timer = PICTimer.getTimer()
    timer.startTimer()

    # for _ in range(10):
    #     x = torch.rand(1920, 1080, 3, 2, dtype=torch.float32)
    #     torch.cuda.synchronize()
    #     timer.showTime('rand')
    #     x = x.to('cuda:0')
    #     torch.cuda.synchronize()
    #     timer.showTime('move')

    back = cv2.imread('/data/FastMVSNet/lab/Rectified/scan1_b/rect_001_3_r5000.png')
    fore = cv2.imread('/data/FastMVSNet/lab/Rectified/scan1_f/rect_001_3_r5000.png')
    back = cv2.pyrDown(back)
    fore = cv2.pyrDown(fore)

    torch.cuda.synchronize()
    timer.showTime('read')
    back = torch.tensor(back, dtype=torch.float32).to('cuda:0')
    fore = torch.tensor(fore, dtype=torch.float32).to('cuda:0')
    torch.cuda.synchronize()
    timer.showTime('move')

    timer.summary()

#test_data_copy()


def crop_image_to_align_fastmvs():
    for idx in [1, 2, 3, 4, 5]:
        back = cv2.imread('/data/FastMVSNet/lab/Rectified/scan1_crop_b/rect_00{}_3_r5000.png'.format(idx))
        back = back[16:-16, :, :]
        fore = cv2.imread('/data/FastMVSNet/lab/Rectified/scan1/rect_00{}_3_r5000.png'.format(idx))
        fore = fore[16:-16, :, :]

        cv2.imwrite('/home/wjk/workspace/PyProject/FastMVSNet/tools/image/back/{}.png'.format(idx), back)
        cv2.imwrite('/home/wjk/workspace/PyProject/FastMVSNet/tools/image/fore/{}.png'.format(idx), fore)

        # croped_fore = cv2.imread('/data/FastMVSNet/lab/lab_crop/scan1/0000000{}.jpg'.format(idx - 1))
        # cv2.imshow('1', back)
        # cv2.waitKey()
        # cv2.imshow( '1', fore)
        # cv2.waitKey()
        # cv2.imshow('1', croped_fore)
        # cv2.waitKey()

        ###visualize the confidence map
        # H, W, _ = fore.shape
        # prob_tmp = load_pfm('/data/FastMVSNet/{}/{}/scan1/0000000{}_init_prob.pfm'.format('lab', 'lab_crop', idx - 1))
        # prob_tmp = prob_tmp[0]
        # prob_tmp = cv2.resize(prob_tmp, [W, H])
        # prob_tmp[prob_tmp < 0.9] = 0
        # prob_tmp = (prob_tmp * 255).astype('uint8')
        # cv2.imshow('1', prob_tmp)
        # cv2.waitKey()

#crop_image_to_align_fastmvs()


def read_blending_weights():
    weights = np.load('/home/pic/downloads/FastMVSNet/outputs/blending_weights.npy', allow_pickle=True)#.item()
    weight = torch.tensor(weights[:,:,:,0]).permute(2,0,1)
    weight = weight.view(4, 1, -1)
    weight = weight.unsqueeze(0).unsqueeze(4)

    J = torch.ones(1,4,48,15360,1)
    J = J * weight
    J = J.permute(0, 3, 1, 2, 4).contiguous().view(-1, 48 * (5 - 1), 1)
    # 15360, 192, 1
    print(1)

#read_blending_weights()

def teaRoom_build():
    abs_path = '/data/GoPro/videos/teaRoom/sequence/video/'
    out_path = '/data/FastMVSNet/teaRoom/Rectified/'
    for idx in range(1,501):
        for view in range(1,6):
            src_path = abs_path + '{}-{}.png'.format(view, idx)
            dst_path = out_path + 'scan{}'.format(idx)
            if os.path.exists(dst_path) is False:
                os.mkdir(dst_path)
            img_dst_path = dst_path + '/rect_00{}_3_r5000.png'.format(view)
            img = cv2.imread(src_path)
            cv2.imwrite(img_dst_path, img)

#teaRoom_build()

def vis_mask():
    path = '/home/wjk/workspace/PyProject/FastMVSNet/tools/image/mask/5.png'
    mask = cv2.imread(path)
    mask[mask < 100] = 0
    cv2.imshow('mask',mask)
    cv2.waitKey()

#vis_mask()

def vis_pfm():
    path = '/data/GoPro/videos/teaRoom/sequence/depth/0003/0005.pfm'
    depth = load_pfm(path)[0]
    depth[depth < 0.6] = 0.6
    maxn = np.max(depth)
    minn = np.min(depth)
    depth = (depth - minn) / (maxn - minn) * 255
    depth = depth.astype('uint8')
    cv2.imshow('depth', depth)
    cv2.waitKey()

vis_pfm()
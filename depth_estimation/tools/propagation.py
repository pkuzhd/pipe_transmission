import sys
import os
# sys.path.insert(1, os.path.dirname(__file__)+'/..')

from tools.isb_filter import ISB_Filter
from tools.utils import rgb2yCbCr
import torch
import numpy as np
import cv2
from tools.g3d_utils import *
from fastmvsnet.utils.io import mkdir, write_cam_dtu, write_pfm

def vis(img, title = 'vis', cuda = False, norm = False):
    if cuda:
        img = img.cpu().numpy()
    # img = np.log(img)
    #img = 1.0 / img
    if norm:
        maxs = np.max(img)
        img[img < 0] = 0.5
        mins = 0.5#np.min(img)
        img = 255.0 * (img - mins) / (maxs - mins)
        img = img.astype('uint8')
        #img = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
    cv2.imshow(title, img)
    cv2.waitKey()

def vis_pcd(img, depth, intr, extr, idx):
    img_dict, depth_dict, intr_dict, extr_dict = {}, {}, {}, {}
    # depth_tmp = cv2.resize(depth_tmp, [W, H])
    # depth_tmp[depth_tmp < 1.25] = 0
    depth_dict[idx] = depth
    H, W = depth.shape
    rgb_tmp = img
    rgb_tmp = cv2.resize(rgb_tmp, [W, H])
    img_dict[idx] = rgb_tmp

    intr_dict[idx] = intr
    extr_dict[idx] = extr

    pcd_this_frame = vis_pcd_gpu(img_dict, depth_dict, intr_dict, extr_dict, 'cuda:0', vis=False)
    pcd_this_frame = pcd_this_frame.to_legacy()
    o3d.visualization.draw_geometries([pcd_this_frame])

def warping_propagation(rgbd_raw = 'rgbd/rgbd_lab.npy', rgbd_checked = 'rgbd/rgbd_lab_warped.npy', output = 'rgbd/rgbd_lab_propagated.npy', maxd = 1.6, mind = 1.0, pfm_path = None):
    sigma_i = 20
    sigma_s = 25

    img = None
    depth = None
    device = 'cuda:0'
    ans = np.load(rgbd_raw, allow_pickle=True).item()
    imgs = ans['rgb']
    ans = np.load(rgbd_checked, allow_pickle=True).item()
    depths = ans['depth']
    masks = ans['mask']
    intrs = ans['intr']
    extrs = ans['extr']

    for idx in range(len(imgs)):
        H, W, C = imgs[idx].shape
        guide = rgb2yCbCr(torch.tensor(imgs[idx], dtype=torch.float32).to(device)).type(torch.uint8)
        print(guide)
        # if pfm_path is None:
        #     vis(imgs[idx].copy(), title='rgb', norm=False, cuda=False)

        distance_filter = ISB_Filter(1, [W, H], 'cuda:0', '')

        distance_map = depths[idx]
        mask_map = masks[idx]
        #vis(distance_map, title='depth', norm=True, cuda=False)
        distance_map[distance_map > maxd] = maxd
        distance_map[distance_map < mind] = mind
        # vis(distance_map.copy(), title='depth', norm=True, cuda=False)

        # mask_random = np.random.rand(distance_map.shape[0], distance_map.shape[1]).astype(np.f loat32)
        # mask_random[mask_random < 0.3] = -1
        # mask_random[mask_random > 0] = 1
        # distance_map = distance_map * mask_random

        mask_map = mask_map.astype(np.float32)
        distance_map[mask_map < mind - 0.1] *= -1.0

        if pfm_path is None:
            vis(distance_map.copy(), title='After Consistency Check', norm=True, cuda=False)
            vis_pcd(imgs[idx], distance_map, intrs[idx].numpy(), extrs[idx].numpy(), idx)

        distance_map = np.expand_dims(distance_map, axis=0)
        distance_map = torch.from_numpy(distance_map).to(device)
        filtered_distance, _ = distance_filter.apply(guide.clone(), distance_map.clone(), sigma_i / 2,
                                                             sigma_s / 2, 2)
        distance_map = filtered_distance[0]
        ans['depth'][idx] = distance_map.cpu().numpy()

        if pfm_path is None:
            vis(distance_map.clone(), title='After Propagation', norm=True, cuda=True)
            vis_pcd(imgs[idx], distance_map.cpu().numpy(), intrs[idx].numpy(), extrs[idx].numpy(), idx)

        if pfm_path is not None:
            write_pfm(pfm_path + '/{:04d}.pfm'.format(idx+1), distance_map.cpu().numpy())
    if pfm_path is None:
        np.save(output, ans)


def warping_propagation_singleframe(imgs_tensor, checked_depth_tensor, checked_masks_tensor, intrs_tensor, extrs_tensor, output = 'rgbd/rgbd_lab_propagated.npy', maxd = 1.6, mind = 0.5, pfm_path = None, device='cuda:0'):
    sigma_i = 20
    sigma_s = 25

    img = None
    depth = None

    imgs = imgs_tensor[0]
    imgs = (imgs.permute((0,2,3,1)))*255.0
    depths = checked_depth_tensor[0]
    masks = checked_masks_tensor[0]
    intrs = intrs_tensor
    extrs = extrs_tensor

    ans = torch.ones_like(depths)

    for idx in range(imgs.shape[0]):
        H, W, C = imgs[idx].shape
        guide = rgb2yCbCr(imgs[idx]).type(torch.uint8)
        # if pfm_path is None:
        #     vis(imgs[idx].clone(), title='rgb', norm=False, cuda=True)
        distance_filter = ISB_Filter(1, [W, H], device, '')

        distance_map = depths[idx]
        mask_map = masks[idx]

        #vis(distance_map, title='depth', norm=True, cuda=False)
        distance_map[distance_map > maxd] = maxd
        distance_map[distance_map < mind] = mind
        # distance_map[distance_map < mind] = maxd
        # vis(distance_map.copy(), title='depth', norm=True, cuda=False)

        # mask_random = np.random.rand(distance_map.shape[0], distance_map.shape[1]).astype(np.f loat32)
        # mask_random[mask_random < 0.3] = -1
        # mask_random[mask_random > 0] = 1
        # distance_map = distance_map * mask_random

        # mask_map = mask_map.astype(np.float32)
        mask_map = mask_map.float()
        distance_map[mask_map < mind - 0.1] *= -1.0

        # if pfm_path is None:
        #     vis(distance_map.copy(), title='After Consistency Check', norm=True, cuda=False)
        #     vis_pcd(imgs[idx], distance_map, intrs[idx].numpy(), extrs[idx].numpy(), idx)

        # distance_map = np.expand_dims(distance_map, axis=0)
        # distance_map = torch.from_numpy(distance_map).to(device)
        distance_map = distance_map.unsqueeze(0)
        # print(f"distance_map: {distance_map.shape}", distance_map)

        filtered_distance, _ = distance_filter.apply(guide.clone(), distance_map.clone(), sigma_i / 2,
                                                             sigma_s / 2, 2)
        distance_map = filtered_distance[0]
        ans[idx] = distance_map
        # print(f"applied distance map: {distance_map.shape}", distance_map)

    ans = ans.unsqueeze(0)
    return ans


def random_propagation():
    sigma_i = 20
    sigma_s = 25

    img = None
    depth = None
    device = 'cuda:0'
    ans = np.load('rgbd/rgbd_by_fastMVSnet.npy', allow_pickle=True).item()
    imgs = ans['rgb']
    depths = ans['depth']

    idx = 1
    H, W, C = imgs[idx].shape
    guide = rgb2yCbCr(torch.tensor(imgs[idx], dtype=torch.float32).to(device)).type(torch.uint8)
    vis(imgs[idx].copy(), title='rgb', norm=False, cuda=False)

    distance_filter = ISB_Filter(1, [W, H], 'cuda:0', '')

    distance_map = depths[idx]
    #vis(distance_map, title='depth', norm=True, cuda=False)
    distance_map[distance_map > 1.6] = 1.6
    distance_map[distance_map < 1.0] = 1.0
    vis(distance_map.copy(), title='depth', norm=True, cuda=False)

    mask_random = np.random.rand(distance_map.shape[0], distance_map.shape[1]).astype(np.float32)
    mask_random[mask_random < 0.9] = -1
    mask_random[mask_random > 0] = 1
    distance_map = distance_map * mask_random
    vis(distance_map.copy(), title='mask_random', norm=True, cuda=False)

    distance_map = np.expand_dims(distance_map, axis=0)
    distance_map = torch.from_numpy(distance_map).to(device)
    filtered_distance, _ = distance_filter.apply(guide.clone(), distance_map.clone(), sigma_i / 2,
                                                         sigma_s / 2, 2)
    distance_map = filtered_distance[0]
    vis(distance_map.clone(), title='filtered', norm=True, cuda=True)

# warping_propagation()
#random_propagation()
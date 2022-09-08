import cv2
import open3d as o3d
import os
import cv2 as cv
import numpy as np
from fastmvsnet.utils.io import *
from propagation import warping_propagation

def convert2npy(output_path, flow, scene, config, frame, num_view):
    img, depth, intr, extr = {}, {}, {}, {}
    result = {'rgb': {}, 'depth': {}, 'intr': {}, 'extr': {}}
    for ref_idx in range(num_view):
        depth_tmp = load_pfm('/data/FastMVSNet/{}/{}/{}/0000000{}_{}.pfm'.format(scene, config, frame, ref_idx, flow))
        depth_tmp = depth_tmp[0]
        H, W = depth_tmp.shape
        prob_tmp = load_pfm('/data/FastMVSNet/{}/{}/{}/0000000{}_init_prob.pfm'.format(scene, config, frame, ref_idx))
        prob_tmp = prob_tmp[0]
        prob_tmp = cv2.resize(prob_tmp, [W, H])
        
        depth_tmp[prob_tmp < 0.55] = 0
        # mask

        rgb_tmp = cv2.imread('/data/FastMVSNet/{}/{}/{}/0000000{}.jpg'.format(scene, config, frame, ref_idx))
        rgb_tmp = cv2.resize(rgb_tmp, [W, H])
        img[ref_idx] = rgb_tmp
        #depth_tmp = cv2.resize(depth_tmp, [W, H])
        #depth_tmp[depth_tmp < 1.25] = 0

        depth[ref_idx] = depth_tmp
        cam = load_cam_dtu(open('/data/FastMVSNet/{}/{}/{}/cam_0000000{}_{}.txt'.format(scene, config, frame, ref_idx, flow)))
        extrinsic = cam[0:4][0:4][0]
        intrinsic = cam[0:4][0:4][1]
        intrinsic = intrinsic[0:3, 0:3]

        intr[ref_idx] = intrinsic
        extr[ref_idx] = extrinsic

    result['rgb'] = img
    result['depth'] = depth
    result['intr'] = intr
    result['extr'] = extr
    np.save(output_path, result)


def depth_map_fusion(input_path, output_path, device, pixel_thre, depth_thre, absolute_depth, view_thre, is_warping=False):
    if is_warping is False:
        cmd = 'pic-consis-checker'
        # 校验+点云（out）
    else:
        cmd = 'pic-consis-projector'
        # 校验+点云+投到5个RGBD

    cmd = cmd + ' --npy_url ' + input_path # 
    cmd = cmd + ' --output_url ' + output_path # 输出地址
    cmd = cmd + ' --device ' + device
    cmd = cmd + ' --pixel_thre ' + str(pixel_thre)  
    cmd = cmd + ' --depth_thre ' + str(depth_thre)
    cmd = cmd + ' --absolute_depth ' + str(absolute_depth) 
    cmd = cmd + ' --view_thre ' + str(view_thre)    # 一致性视角
    print(cmd)
    os.system(cmd)

    return

is_propagation = False#True#
pixel_threshold = 1.0
depth_threshold = 0.01#10
absolute_depth = False
num_consistent = 3
device = 'cpu'#'cuda:1'
flow = 'flow1'
scene = 'lab'#'teaRoom'#'dtu'##'lab'#'gym'
config = 'lab_crop'#'teaRoom'#'dtu_5views'##'lab_crop'#'gym'
frame = 'scan1'
maxd=1.6#1000#
mind=1.0#0.6#400#
num_view = 5#10#

rgbd_cam_path = 'rgbd/rgbd_{}.npy'.format(scene)
warped_rgbd_cam_path = 'rgbd/rgbd_{}_warped.npy'.format(scene)
propagated_rgbd_cam_path = 'rgbd/rgbd_{}_propagated.npy'.format(scene)
pcd_path = 'pcd/pcd_{}.ply'.format(scene)

convert2npy(rgbd_cam_path, flow, scene, config, frame, num_view)
if is_propagation is False:
    depth_map_fusion(rgbd_cam_path, pcd_path, device, pixel_threshold, depth_threshold, absolute_depth, num_consistent)
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd])
else:
    depth_map_fusion(rgbd_cam_path, warped_rgbd_cam_path, device, pixel_threshold, depth_threshold, absolute_depth, num_consistent, True)
    warping_propagation(rgbd_raw=rgbd_cam_path, rgbd_checked=warped_rgbd_cam_path, output=propagated_rgbd_cam_path, maxd=maxd, mind=mind)

# #### Check and Propagation for video frames
# for idx in range(101, 201):
#     frame = 'scan{}'.format(idx)
#     pfm_path = '/data/GoPro/videos/teaRoom/sequence/depth/{:04d}'.format(idx)
#     if not os.path.exists(pfm_path):
#         os.mkdir(pfm_path)
#     convert2npy(rgbd_cam_path, flow, scene, config, frame, 5)
#     if is_propagation is False:
#         depth_map_fusion(rgbd_cam_path, pcd_path, device, pixel_threshold, depth_threshold, absolute_depth,
#                          num_consistent)
#         pcd = o3d.io.read_point_cloud(pcd_path)
#         o3d.visualization.draw_geometries([pcd])
#     else:
#         depth_map_fusion(rgbd_cam_path, warped_rgbd_cam_path, device, pixel_threshold, depth_threshold, absolute_depth,
#                          num_consistent, True)
#         warping_propagation(rgbd_raw=rgbd_cam_path, rgbd_checked=warped_rgbd_cam_path, output=propagated_rgbd_cam_path,
#                             maxd=maxd, mind=mind, pfm_path=pfm_path)
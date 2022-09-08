"""
GTA-IM Dataset
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt

def vis_pcd(rgb, depth, intr, extr, vis=True):
    global_pcd = o3d.geometry.PointCloud()
    # use nearby RGBD frames to create the environment point cloud
    for idx in rgb.keys():
        depth[idx][depth[idx] < 2.0] = 0
        depth_raw = o3d.geometry.Image(depth[idx].astype(np.float32))
        color_raw = cv2.cvtColor(rgb[idx], cv2.COLOR_BGR2RGB)
        color_raw = o3d.geometry.Image(color_raw)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw,
            depth_raw,
            depth_scale=1.0,
            depth_trunc=10.0,#15.0,
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsic(
                    depth[idx].shape[1], depth[idx].shape[0], intr[idx][0,0], intr[idx][1,1], intr[idx][0,2], intr[idx][1,2]
                )
            ),
        )
        depth_pts = np.asarray(pcd.points)

        depth_pts_aug = np.hstack(
           [depth_pts, np.ones([depth_pts.shape[0], 1])]
        )
        depth_pts = depth_pts_aug.dot(extr[idx][0:3,].T)
        pcd.points = o3d.utility.Vector3dVector(depth_pts)

        global_pcd.points.extend(pcd.points)
        global_pcd.colors.extend(pcd.colors)
    if vis:
        vis_list = [global_pcd]
        o3d.visualization.draw_geometries(vis_list, front=[0.5297, -0.1873, -0.8272],
                                          lookat=[2.0712, 2.0312, 1.7251],
                                          up=[-0.0558, -0.9809, 0.1864],
                                          zoom=0.47)
    return global_pcd

def vis_pcd_gpu(rgb, depth, intr, extr, device, vis=True):
    device = o3d.core.Device(str(device))
    global_pcd = None
    # use nearby RGBD frames to create the environment point cloud
    for idx in rgb.keys():
        #depth[idx][depth[idx] > 7.0] = 0
        depth_raw = o3d.t.geometry.Image(depth[idx].astype(np.float32))
        #depth_raw = depth_raw.to(device)
        color_raw = cv2.cvtColor(rgb[idx], cv2.COLOR_BGR2RGB)
        color_raw = o3d.t.geometry.Image(color_raw.astype(np.float32) / 255.)
        #color_raw = color_raw.to(device)

        extrinsic = extr[idx]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth[idx].shape[1], depth[idx].shape[0], intr[idx][0, 0], intr[idx][1, 1], intr[idx][0, 2],
            intr[idx][1, 2]
        )

        # moving extrinsic and intrinsic to CUDA
        extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)
        intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)

        rgbd_image = o3d.t.geometry.RGBDImage(
            color_raw,
            depth_raw,
        ).to(device)
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics=intrinsic,
            extrinsics=extrinsic,
            depth_scale = 1.,
            depth_max = 20.,
        )
        if global_pcd is None:
            global_pcd = pcd
        else:
            global_pcd = global_pcd.append(pcd)

    if vis:
        o3d.visualization.draw([global_pcd])
    return global_pcd

def vis_mesh(rgb, depth, intr, extr):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for idx in rgb.keys():
        # depth[idx][depth[idx] < 2.0] = 0
        depth_raw = o3d.geometry.Image(depth[idx].astype(np.float32))
        color_raw = cv2.cvtColor(rgb[idx], cv2.COLOR_BGR2RGB)
        color_raw = o3d.geometry.Image(color_raw)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw,
            depth_raw,
            depth_scale=1.0,
            depth_trunc=10.0,  # 15.0,
            convert_rgb_to_intensity=False,
        )

        volume.integrate(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsic(
                    depth[idx].shape[1], depth[idx].shape[0], intr[idx][0, 0], intr[idx][1, 1], intr[idx][0, 2],
                    intr[idx][1, 2]
                )
            ),
            np.linalg.inv(extr[idx])
        )

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh],
                                      front=[0.5297, -0.1873, -0.8272],
                                      lookat=[2.0712, 2.0312, 1.7251],
                                      up=[-0.0558, -0.9809, 0.1864],
                                      zoom=0.47)
    #o3d.io.write_triangle_mesh('D:\\mesh_out\\mesh.ply', mesh)

def read_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d.visualization.draw_geometries([mesh],
                                      front=[0.5297, -0.1873, -0.8272],
                                      lookat=[2.0712, 2.0312, 1.7251],
                                      up=[-0.0558, -0.9809, 0.1864],
                                      zoom=0.47)

def TSDF_RayCasting(vbg, intrinsic, extrinsic, width, height):
    start = time.time()
    # TSDF ray_cast
    block_coords = vbg.hashmap().key_tensor()
    out_img = vbg.ray_cast(block_coords=block_coords, intrinsic=intrinsic, extrinsic=extrinsic,
                           width=width, height=height, render_attributes=['color'],
                           depth_scale=1.0, depth_min=0.55, depth_max=15.0, weight_threshold=1.0,
                           trunc_voxel_multiplier=8.0, range_map_down_factor=32)
    # plt.imshow(out_img['color'].cpu().numpy())
    # plt.show()
    result = out_img['color'].cpu().numpy()
    dt = time.time() - start
    print('Finished Ray Casting in {} seconds'.format(dt))
    result = (result * 255).astype('uint8')
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # cv2.imshow("warped", result)
    # cv2.waitKey()
    return result

def load_trinsic(intrinsic, extrinsic):
    intrinsic = o3d.core.Tensor(intrinsic, o3d.core.Dtype.Float64)
    extrinsic = o3d.core.Tensor(extrinsic,o3d.core.Dtype.Float64)
    return intrinsic, extrinsic

def PCD2IMG(pcd, intrinsic, extrinsic, width, height):
    out_img = pcd.project_to_rgbd_image(width=width, height=height, intrinsics=intrinsic,
                                        extrinsics=extrinsic, depth_scale=1.0, depth_max=30.0)
    out_img = (np.asarray(out_img.cpu().color) * 255).astype('uint8')
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("warped", out_img)
    cv2.waitKey()


def integrate(rgb, depth, intr, extr, device, only_depth = False):
    start = time.time()
    device = o3d.core.Device(str(device))
    if only_depth is False:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=8.0 / 512.0,
            block_resolution=16, # 16 is too large
            block_count=50000,
            device=device)
    else:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight'),
            attr_dtypes=(o3c.float32, o3c.float32),
            attr_channels=((1), (1)),
            voxel_size=8.0 / 512.0,
            block_resolution=4,  # 16 is too large
            block_count=50000,
            device=device)

    for idx in rgb.keys():
        print('Integrating frame {}'.format(idx))
        #depth[idx][depth[idx] < 2.0] = 0
        depth[idx][depth[idx] > 15.0] = 0
        t_todevice = time.time()
        depth_raw = o3d.t.geometry.Image(depth[idx].astype(np.float32))
        depth_raw = depth_raw.to(device)

        if only_depth is False:
            color_raw = cv2.cvtColor(rgb[idx], cv2.COLOR_BGR2RGB)
            color_raw = o3d.t.geometry.Image(color_raw.astype(np.float32) / 255.)
            color = color_raw.to(device)

        extrinsic = extr[idx]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
                depth[idx].shape[1], depth[idx].shape[0], intr[idx][0, 0], intr[idx][1, 1], intr[idx][0, 2],
                intr[idx][1, 2]
        )

        #moving extrinsic and intrinsic to CUDA
        extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)
        intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)
        if idx == 4:
            project_width, project_height = depth[idx].shape[1], depth[idx].shape[0]
            project_extrinsic = extrinsic
            project_intrinsic = intrinsic

        print('todevice:', time.time() - t_todevice)

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth_raw, intrinsic, extrinsic, depth_scale=1.0,
            depth_max=15.0)

        if only_depth is False:
            vbg.integrate(block_coords=frustum_block_coords, depth=depth_raw, color=color,
                          intrinsic=intrinsic, extrinsic=extrinsic,
                          depth_scale=1.0, depth_max=15.0, trunc_voxel_multiplier = 8.0)
        else:
            vbg.integrate(block_coords=frustum_block_coords, depth=depth_raw,
                          intrinsic=intrinsic, extrinsic=extrinsic,
                          depth_scale=1.0, depth_max=15.0, trunc_voxel_multiplier = 8.0)

    dt = time.time() - start
    print('Finished integrating in {} seconds'.format(dt))
    # print('Saving to {}...'.format())
    # vbg.save('lab.npz')
    # print('Saving finished')

    # #TSDF ray_cast
    # img = TSDF_RayCasting(vbg = vbg, intrinsic = project_intrinsic, extrinsic = project_extrinsic, width = project_width, height = project_height)

    # start = time.time()
    # pcd = vbg.extract_point_cloud(-1,-1)#.to_legacy()
    # PCD2IMG(pcd = pcd, intrinsic = project_intrinsic, extrinsic = project_extrinsic, width = project_width, height = project_height)
    # dt = time.time() - start
    # print('Finished extract_point_cloud in {} seconds'.format(dt))
    # o3d.visualization.draw_geometries([pcd.to_legacy()])

    # start = time.time()
    # mesh = vbg.extract_triangle_mesh(0, -1)#.to_legacy()
    # dt = time.time() - start
    # print('Finished extract_triangle_mesh in {} seconds'.format(dt))
    # o3d.visualization.draw_geometries([mesh.to_legacy()])

    return vbg


def warp_pcd(rgb, depth, intr, extr, device, viewpoint):
    start = time.time()
    device = o3d.core.Device(str(device))
    global_pcd = None
    # use nearby RGBD frames to create the environment point cloud
    for idx in rgb.keys():
        #depth[idx][depth[idx] < 2.0] = 0
        depth_raw = o3d.t.geometry.Image(depth[idx].astype(np.float32))
        #depth_raw = depth_raw.to(device)
        color_raw = cv2.cvtColor(rgb[idx], cv2.COLOR_BGR2RGB)
        color_raw = o3d.t.geometry.Image(color_raw.astype(np.float32) / 255.)
        #color_raw = color_raw.to(device)

        extrinsic = extr[idx]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth[idx].shape[1], depth[idx].shape[0], intr[idx][0, 0], intr[idx][1, 1], intr[idx][0, 2],
            intr[idx][1, 2]
        )

        # moving extrinsic and intrinsic to CUDA
        extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)
        intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)
        if idx == viewpoint:
            project_weight, project_height = depth[idx].shape[1], depth[idx].shape[0]
            project_extrinsic = extrinsic
            project_intrinsic = intrinsic

        rgbd_image = o3d.t.geometry.RGBDImage(
            color_raw,
            depth_raw,
        ).to(device)
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics=intrinsic,
            extrinsics=extrinsic,
            depth_scale = 1.,
            depth_max = 20.,
        )
        if global_pcd is None:
            global_pcd = pcd
        else:
            global_pcd = global_pcd.append(pcd)

    #o3d.visualization.draw([global_pcd])
    print(project_extrinsic)
    out_img = global_pcd.project_to_rgbd_image(width=project_weight, height=project_height, intrinsics=project_intrinsic,
                                        extrinsics=project_extrinsic, depth_scale=1.0, depth_max=20.0)
    out_img = (np.asarray(out_img.cpu().color) * 255).astype('uint8')
    print('Finished DIBR in {} seconds'.format(time.time()-start))
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    cv2.imshow("warped", out_img)
    cv2.waitKey()
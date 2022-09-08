import cv2
import open3d as o3d
import os
import cv2 as cv
import numpy as np
from fastmvsnet.utils.io import *
from g3d_utils import *
#
# pcd = o3d.io.read_point_cloud('/data/FastMVSNet/SampleSet/MVS_Data/dtu/LANCZOS4_3ITER_flow3_ip0.2_fp0.1_d10.1_nc0/scan1_ip0.2_fp0.1_d10.1_nc0_LANCZOS4.ply')
# o3d.visualization.draw_geometries([pcd])

def fastMVSnet_vis_crop(scene, config, frame):
    for ref_idx in [0, 1, 2, 3, 4]:
        for flow in ['flow3']:
            img, depth, intr, extr = {}, {}, {}, {}
            depth_tmp = load_pfm('/data/FastMVSNet/{}/{}/{}/0000000{}_{}.pfm'.format(scene, config, frame, ref_idx, flow))
            depth_tmp = depth_tmp[0]
            # depth_tmp = cv2.resize(depth_tmp, [W, H])
            # depth_tmp[depth_tmp < 1.25] = 0
            depth[ref_idx] = depth_tmp
            H, W = depth_tmp.shape
            rgb_tmp = cv2.imread('/data/FastMVSNet/{}/{}/{}/0000000{}.jpg'.format(scene, config, frame, ref_idx))
            rgb_tmp = cv2.resize(rgb_tmp, [W, H])
            img[ref_idx] = rgb_tmp

            cam = load_cam_dtu(open('/data/FastMVSNet/{}/{}/{}/cam_0000000{}_{}.txt'.format(scene, config, frame, ref_idx, flow)))
            extrinsic = cam[0:4][0:4][0]
            intrinsic = cam[0:4][0:4][1]
            intrinsic = intrinsic[0:3, 0:3]
            intr[ref_idx] = intrinsic
            extr[ref_idx] = extrinsic

            pcd_this_frame = vis_pcd_gpu(img, depth, intr, extr, 'cuda:0', vis=False)
            pcd_this_frame = pcd_this_frame.to_legacy()
            o3d.visualization.draw_geometries([pcd_this_frame])

pcd = o3d.io.read_point_cloud('/home/wjk/workspace/PyProject/FastMVSNet/tools/pcd/pcd_lab_coarse_feature_warping_3scale.ply')
o3d.visualization.draw_geometries([pcd])
pcd = o3d.io.read_point_cloud('/home/wjk/workspace/PyProject/FastMVSNet/tools/pcd/pcd_lab_flow_feature_warping_3scale.ply')
o3d.visualization.draw_geometries([pcd])
pcd = o3d.io.read_point_cloud('/home/wjk/workspace/PyProject/FastMVSNet/tools/pcd/pcd_lab_coarse_feature_warping.ply')
o3d.visualization.draw_geometries([pcd])
pcd = o3d.io.read_point_cloud('/home/wjk/workspace/PyProject/FastMVSNet_custom/FastMVSNet/tools/pcd/pcd_lab_coarse_warping_feature.ply')
o3d.visualization.draw_geometries([pcd])
#fastMVSnet_vis_crop('dtu', 'dtu_5views', 'scan6')
fastMVSnet_vis_crop('lab', 'lab_crop', 'scan1')
#pcd = o3d.io.read_point_cloud('/home/wjk/workspace/PyProject/FastMVSNet/tools/pcd/rgbd_crop_by_fastMVSnet_0.26_5view.ply')
pcd = o3d.io.read_point_cloud('/home/wjk/workspace/PyProject/MVS2D/pcd/rgbd_crop_by_fastMVSnet_0.26.ply')
o3d.visualization.draw_geometries([pcd])

#pcd = o3d.io.read_point_cloud('/data/FastMVSNet/lab/lab_fore/LANCZOS4_3ITER_flow5_ip0.05_fp0.1_d0.8_nc3/scan1_ip0.05_fp0.1_d0.8_nc3_LANCZOS4.ply')
#pcd = o3d.io.read_point_cloud('/data/FastMVSNet/lab/lab_fore/LANCZOS4_3ITER_flow4_ip0.05_fp0.1_d0.8_nc3/scan1_ip0.05_fp0.1_d0.8_nc3_LANCZOS4.ply')
#pcd = o3d.io.read_point_cloud('/data/FastMVSNet/lab/lab_fore/LANCZOS4_3ITER_flow3_ip0.05_fp0.1_d0.8_nc3/scan1_ip0.05_fp0.1_d0.8_nc3_LANCZOS4.ply')
#pcd = o3d.io.read_point_cloud('/data/FastMVSNet/lab/lab_fore/LANCZOS4_3ITER_flow2_ip0.05_fp0.1_d0.8_nc3/scan1_ip0.05_fp0.1_d0.8_nc3_LANCZOS4.ply')
#pcd = o3d.io.read_point_cloud('/data/FastMVSNet/lab/lab_fore/LANCZOS4_3ITER_flow1_ip0.05_fp0.1_d0.8_nc3/scan1_ip0.05_fp0.1_d0.8_nc3_LANCZOS4.ply')
#pcd = o3d.io.read_point_cloud('/data/FastMVSNet/lab/lab_fore/LANCZOS4_3ITER_init_ip0.05_fp0.1_d0.8_nc3/scan1_ip0.05_fp0.1_d0.8_nc3_LANCZOS4.ply')
#o3d.visualization.draw_geometries([pcd])

#pcd = o3d.io.read_point_cloud('/data/FastMVSNet/lab/lab_fore/LANCZOS4_3ITER_flow1_ip0.05_fp0.1_d0.8_nc3/scan1_ip0.05_fp0.1_d0.8_nc3_LANCZOS4.ply')
#o3d.visualization.draw_geometries([pcd])

#pcd = o3d.io.read_point_cloud('/data/FastMVSNet/lab/lab_fore/LANCZOS4_3ITER_flow2_ip0.05_fp0.1_d0.8_nc3/scan1_ip0.05_fp0.1_d0.8_nc3_LANCZOS4.ply')
#o3d.visualization.draw_geometries([pcd])

# pcd = o3d.io.read_point_cloud('/data/FastMVSNet/lab/lab/LANCZOS4_3ITER_flow3_ip0.2_fp0.1_d0.1_nc1/scan1_ip0.2_fp0.1_d0.1_nc1_LANCZOS4.ply')
# o3d.visualization.draw_geometries([pcd])


# pcd = o3d.io.read_point_cloud('/data/FastMVSNet/talking/talking/LANCZOS4_3ITER_flow3_ip0.2_fp0.1_d0.1_nc1/scan1_ip0.2_fp0.1_d0.1_nc1_LANCZOS4.ply')
# o3d.visualization.draw_geometries([pcd])


pcd.estimate_normals()


mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.005)
o3d.visualization.draw_geometries([mesh])

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
o3d.visualization.draw_geometries([mesh[0]])


# estimate radius for rolling ball
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 5.5 * avg_dist

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))
print(mesh.get_surface_area())
o3d.visualization.draw_geometries([mesh])
# o3d.visualization.draw_geometries([mesh], window_name='Open3D downSample', width=800, height=600, left=50,
#                                   top=50, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True,)
# # create the triangular mesh with the vertices and faces from open3d
# tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
#                           vertex_normals=np.asarray(mesh.vertex_normals))
#
# trimesh.convex.is_convex(tri_mesh)

def compute_res():
    abs_path = '/home/wjk/workspace/PyProject/Realtime_Depth_Estimation/data/wict/'
    for idx in [1,2,3,4,5]:
        fore_path = os.path.join(abs_path,str(idx),'009-1.png')
        back_path = os.path.join(abs_path,str(idx),'010-1.png')
        fore_img = cv.imread(fore_path).astype('float64')
        back_img = cv.imread(back_path).astype('float64')
        res = fore_img - back_img
        res = np.abs(res)
        res = np.sum(res, axis=2) / 3
        fore_img[res < 25] = 0
        #res = res.astype('uint8')
        fore_img = fore_img.astype('uint8')
        # cv.imshow('res', fore_img)
        # cv.waitKey()
        cv.imwrite(os.path.join(abs_path,str(idx),'fore.png'), fore_img)


def vis_pcd():
    depth = np.load('/data/depth_zhd/depth.npy')
    depth = depth / 400 * 20
    depth_raw = o3d.geometry.Image(depth.astype(np.float32))
    color_raw = cv.imread('/home/wjk/workspace/PyProject/MVS2D/demo/lab/rgb_b/000000.png')
    color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
    color_raw = cv.resize(color_raw, [960, 540])
    color_raw = o3d.geometry.Image(color_raw)

    intr = np.array([[1839.06, 0.0, 1922.53],[0.0, 1838.96, 1063.40],[0.0, 0.0, 1.0]])
    intr[0:2] = intr[0:2] * 960 / 3840

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
                depth.shape[1], depth.shape[0], intr[0,0], intr[1,1], intr[0,2], intr[1,2]
            )
        ),
    )
    o3d.visualization.draw_geometries([pcd])
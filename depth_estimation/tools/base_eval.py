import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import open3d as o3d
from time import time
from random import seed
import argparse
import scipy.io as sio
from sklearn.neighbors import KDTree

seed(5)
np.random.seed(5)
milliseconds = int(time() * 1000)

# argument parsing
parse = argparse.ArgumentParser(description="Depth Map Fusion Deep Network evaluation.")

parse.add_argument("-m", "--method", default="colmap", type=str, help="Method name (e.x. colmap).")
parse.add_argument("-d", "--data_path", default="/data/FastMVSNet/SampleSet/MVS_Data", type=str, help="Path to the DTU evaluation data.")
parse.add_argument("-r", "--results_path", default="../../Results", type=str, help="Output results path where the output evaluation metrics will be stored.")
parse.add_argument("-l", "--light_setting", default="l3", type=str, help="DTU light setting.")
parse.add_argument("-p", "--representation", default="Points", type=str, help="Data representation (Points/Surface).")
parse.add_argument("-e", "--eval_list", default="6", type=str, help="Scene evaluation list following the format '#,#,#,#' (e.x. '1,9,23,77,114') COMMA-SEPARATED, NO-SPACES.")

ARGS = parse.parse_args()


def correct_round(n):
    return np.round(n+0.5)

def read_point_cloud(ply_path):
    if(ply_path[-3:] != "ply"):
        print("Error: file {} is not a '.ply' file.".format(ply_path))

    ply = o3d.io.read_point_cloud(ply_path, format="ply")

    return ply

def reduce_points(ply, min_dist):
    points = np.asarray(ply.points)
    n = len(points)
    rand_inds = np.random.permutation(n)
    batch_size = int(4e6)
    index_set = np.ones(n)

    tree = KDTree(points)
    
    step_size = min(batch_size, n)

    for i in range(0,n,step_size):
        batch = rand_inds[i:i+step_size]
        rand_points = np.asarray(points[batch])
        inds = tree.query_radius(rand_points, r=min_dist)

        for i in range(len(inds)):
            ind = batch[i]
            if(index_set[ind]):
                index_set[inds[i]] = 0
                index_set[ind] = 1

    valid_inds = list(np.asarray(np.where(index_set==1), dtype=int).squeeze(0))

    #print("point cloud reduced to {}".format(len(valid_inds)))
    #print("downsample factor: {:0.4f}".format(n/len(valid_inds)))

    return ply.select_by_index(valid_inds)

def build_est_points_filter(ply, min_bound, res, mask):
    points = np.asarray(ply.points).transpose()
    shape = points.shape
    mask_shape = mask.shape
    filt = np.zeros(shape[1])

    min_bound = min_bound.reshape(3,1)
    min_bound = np.tile(min_bound, (1,shape[1]))

    qv = points
    qv = (points - min_bound) / res
    qv = correct_round(qv).astype(int)

    # get all valid points
    in_bounds = np.asarray(np.where( ((qv[0,:]>=0) & (qv[0,:] < mask_shape[0]) & (qv[1,:]>=0) & (qv[1,:] < mask_shape[1]) & (qv[2,:]>=0) & (qv[2,:] < mask_shape[2])))).squeeze(0)
    valid_points = qv[:,in_bounds]

    # convert 3D coords ([x,y,z]) to appropriate flattened coordinate ((x*mask_shape[1]*mask_shape[2]) + (y*mask_shape[2]) + z )
    mask_inds = np.ravel_multi_index(valid_points, dims=mask.shape, order='C')

    # further trim down valid points by mask value (keep point if mask is True)
    mask = mask.flatten()
    valid_mask_points = np.asarray(np.where(mask[mask_inds] == True)).squeeze(0)

    # add 1 to indices where we want to keep points
    filt[in_bounds[valid_mask_points]] = 1

    return filt

def build_gt_points_filter(ply, P):
    points = np.asarray(ply.points).transpose()
    shape = points.shape

    # compute iner-product between points and the defined plane
    Pt = P.transpose()

    points = np.concatenate((points, np.ones((1,shape[1]))), axis=0)
    plane_prod = (Pt @ points).squeeze(0)

    # get all valid points
    filt = np.asarray(np.where((plane_prod > 0), 1, 0))

    return filt

def point_cloud_dist(src_ply, tgt_ply, max_dist, filt):
    # for every point in src, compute distance to target
    dists = src_ply.compute_point_cloud_distance(tgt_ply)
    dists = np.asarray(dists)
    dists = np.clip(dists, 0, 60)

    dists = dists[filt == 1]
    dists = dists[dists <= max_dist]

    num_points = len(dists)

    return num_points, np.mean(dists), np.var(dists), np.median(dists)

def compare_points(est_ply, gt_ply, data_path, max_dist, est_filt, gt_filt):
    # load mask, bounding box, and resolution
    (num_est, mean_acc, var_acc, med_acc) = point_cloud_dist(est_ply, gt_ply, max_dist, est_filt)
    (num_gt, mean_comp, var_comp, med_comp) = point_cloud_dist(gt_ply, est_ply, max_dist, gt_filt)

    return (num_est, num_gt, mean_acc, mean_comp, var_acc, var_comp, med_acc, med_comp)

def main():
    min_dist = 0.2
    max_dist = 20
    
    # convert eval list to integers
    eval_list = ARGS.eval_list.split(',')
    eval_list = [int(e) for e in eval_list]

    num_evals = len(eval_list)

    if (ARGS.representation == "Points"):
        eval_string = "_Eval_IJCV_"
        settings_string = ""
    elif (ARGS.representation == "Surfaces"):
        eval_string = "_SurfEval_Trim_IJCV_"
        settings_string = "_surf_11_trim_8"

    # variables for recording averages
    avg_num_est = 0
    avg_num_gt = 0
    avg_mean_acc = 0.0
    avg_mean_comp = 0.0
    avg_var_acc = 0.0
    avg_var_comp = 0.0
    avg_med_acc = 0.0
    avg_med_comp = 0.0
    avg_dur = 0.0

    for scan_num in eval_list:
        start_total = time()
        print("\nEvaluating scan{:03d}...".format(scan_num))

        # read in matlab bounding box, mask, and resolution
        mask_filename = "ObsMask{}_10.mat".format(scan_num)
        mask_path = os.path.join(ARGS.data_path, "ObsMask", mask_filename)
        data = sio.loadmat(mask_path)
        bounds = np.asarray(data["BB"])
        min_bound = bounds[0,:]
        max_bound = bounds[1,:]
        mask = np.asarray(data["ObsMask"])
        res = int(data["Res"])

        # read in matlab gt plane 
        mask_filename = "Plane{}.mat".format(scan_num)
        mask_path = os.path.join(ARGS.data_path, "ObsMask", mask_filename)
        data = sio.loadmat(mask_path)
        P = np.asarray(data["P"])

        # read in estimated point cloud
        # est_ply_filename = "{}{:03d}_{}{}.ply".format(ARGS.method, scan_num, ARGS.light_setting, settings_string)
        # est_ply_path = os.path.join(ARGS.data_path, ARGS.representation, ARGS.method, est_ply_filename)
        est_ply_path = '/home/wjk/workspace/PyProject/FastMVSNet/tools/pcd/pcd_dtu.ply'
        est_ply = read_point_cloud(est_ply_path)

        # reduce estimated point cloud
        est_ply = reduce_points(est_ply, min_dist)

        # read in ground-truth point cloud
        gt_ply_filename = "stl{:03d}_total.ply".format(scan_num)
        gt_ply_path = os.path.join(ARGS.data_path, "Points", "stl", gt_ply_filename)
        gt_ply = read_point_cloud(gt_ply_path) # already reduced to 0.2mm density, so no downsampling needed

        # build points filter based on input mask
        est_filt = build_est_points_filter(est_ply, min_bound, res, mask)

        # build points filter based on input mask
        gt_filt = build_gt_points_filter(gt_ply, P)

        # compute distances between point clouds
        (num_est, num_gt, mean_acc, mean_comp, var_acc, var_comp, med_acc, med_comp) = \
                compare_points(est_ply, gt_ply, ARGS.data_path, max_dist, est_filt, gt_filt)

        end_total = time()
        dur = end_total-start_total

        # display current evaluation
        print("Num Est: {}".format(int(num_est)))
        print("Num GT: {}".format(int(num_gt)))
        print("Mean Acc: {:0.4f}".format(mean_acc))
        print("Mean Comp: {:0.4f}".format(mean_comp))
        print("Var Acc: {:0.4f}".format(var_acc))
        print("Var Comp: {:0.4f}".format(var_comp))
        print("Med Acc: {:0.4f}".format(med_acc))
        print("Med Comp: {:0.4f}".format(med_comp))
        print("Elapsed time: {:0.3f} s".format(dur))

        # record averages
        avg_num_est     += num_est
        avg_num_gt      += num_gt
        avg_mean_acc    += mean_acc
        avg_mean_comp   += mean_comp
        avg_var_acc     += var_acc
        avg_var_comp    += var_comp
        avg_med_acc     += med_acc
        avg_med_comp    += med_comp
        avg_dur         += dur

    # display average evaluation
    print("\nAveraged evaluation..")
    print("Num Est: {}".format(int(avg_num_est // num_evals)))
    print("Num GT: {}".format(int(avg_num_gt // num_evals)))
    print("Mean Acc: {:0.4f}".format(avg_mean_acc / num_evals))
    print("Mean Comp: {:0.4f}".format(avg_mean_comp / num_evals))
    print("Var Acc: {:0.4f}".format(avg_var_acc / num_evals))
    print("Var Comp: {:0.4f}".format(avg_var_comp / num_evals))
    print("Med Acc: {:0.4f}".format(avg_med_acc / num_evals))
    print("Med Comp: {:0.4f}".format(avg_med_comp / num_evals))
    print("Elapsed time: {:0.3f} s".format(avg_dur / num_evals))


if __name__=="__main__":
    main()

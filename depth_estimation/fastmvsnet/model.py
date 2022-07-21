import time

import cv2
import numpy as np
import collections
from fastmvsnet.utils import io

import torch
import torch.nn as nn
import torch.nn.functional as F
# from picutils import PICTimer

from fastmvsnet.networks import *
from fastmvsnet.functions.functions import get_pixel_grids, get_propability_map
from fastmvsnet.utils.feature_fetcher import FeatureFetcher, FeatureGradFetcher, PointGrad, ProjectUVFetcher


class FastMVSNet(nn.Module):
    def __init__(self,
                 img_base_channels=8,
                 vol_base_channels=8,
                 flow_channels=(64, 64, 16, 1),
                 k=16,
                 ):
        super(FastMVSNet, self).__init__()
        self.k = k

        self.feature_fetcher = FeatureFetcher()
        self.feature_grad_fetcher = FeatureGradFetcher()
        self.point_grad_fetcher = PointGrad()

        self.coarse_img_conv = ImageConv(img_base_channels)
        self.coarse_vol_conv = VolumeConv(img_base_channels * 4, vol_base_channels)
        self.propagation_net = PropagationNet(img_base_channels)
        self.flow_img_conv = ImageConv(img_base_channels)

    def forward(self, data_batch, img_scales, inter_scales, blending, isGN, isTest=False):
        print(f"!forward paras: {img_scales} {inter_scales} {blending} ")
        preds = collections.OrderedDict()
        img_list = data_batch["img_list"]
        cam_params_list = data_batch["cam_params_list"]
        print(f"! model img_list shape/type {img_list.shape}{img_list.dtype}")
        print(f"! model cam_params_list shape/type {cam_params_list.shape}{cam_params_list.dtype}")

        # print(f"! img_tensor : {img_list[0]}")
        # for i in range(5):
        #     io.write_cam_dtu(f"/home/wph/pipe_transmission/depth_estimation/FastMVSNet/cam_para/cam_para{i}.txt", cam_params_list[0][i])

        cam_extrinsic = cam_params_list[:, :, 0, :3, :4].clone()  # (B, V, 3, 4)
        R = cam_extrinsic[:, :, :3, :3]
        t = cam_extrinsic[:, :, :3, 3].unsqueeze(-1)
        R_inv = torch.inverse(R)
        cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()

        if isTest:
            cam_intrinsic[:, :, :2, :3] = cam_intrinsic[:, :, :2, :3] / 4.0

        depth_start = cam_params_list[:, 0, 1, 3, 0]
        depth_interval = cam_params_list[:, 0, 1, 3, 1]
        num_depth = cam_params_list[0, 0, 1, 3, 2].long()

        depth_end = depth_start + (num_depth - 1) * depth_interval

        batch_size, num_view, img_channel, img_height, img_width = list(img_list.size())
        coarse_feature_maps = []

        # torch.cuda.synchronize()
        # timer_estimation = PICTimer.getTimer('estimation')
        # timer_estimation.startTimer()
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :]
            # img_tmp = curr_img[0].permute([1,2,0]).cpu().numpy()
            # minn = np.min(img_tmp)
            # maxn = np.max(img_tmp)
            # print('min: {} and max: {}'.format(minn, maxn))
            # img_tmp = (img_tmp - minn) / (maxn - minn)
            # # img_tmp = img_tmp * 255.0
            # # img_tmp = img_tmp.astype('uint8')
            # cv2.imshow('color', img_tmp)
            # cv2.waitKey()
            curr_feature_map = self.coarse_img_conv(curr_img)["conv2"]
            # torch.save(self.coarse_img_conv.state_dict(), 'img_conv_param.pkl')
            coarse_feature_maps.append(curr_feature_map)
        # torch.cuda.synchronize()
        # timer_estimation.showTime('feature')

        feature_list = torch.stack(coarse_feature_maps, dim=1)

        feature_channels, feature_height, feature_width = list(curr_feature_map.size())[1:]

        depths = []
        for i in range(batch_size):
            depths.append(torch.linspace(depth_start[i], depth_end[i], num_depth, device=img_list.device) \
                          .view(1, 1, num_depth, 1))
            # depths.append((1 / torch.linspace(1 / depth_start[i], 1 / depth_end[i], num_depth, device=img_list.device)) \
            #               .view(1, 1, num_depth, 1))
        # print(f"! depth to estimate: {depths}")
        depths = torch.stack(depths, dim=0)  # (B, 1, 1, D, 1)

        feature_map_indices_grid = get_pixel_grids(feature_height, feature_width)
        # print("before:", feature_map_indices_grid.size())
        feature_map_indices_grid = feature_map_indices_grid.view(1, 3, feature_height, feature_width)[:, :, ::2, ::2].contiguous()
        # print("after:", feature_map_indices_grid.size())
        feature_map_indices_grid = feature_map_indices_grid.view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device)

        ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
        uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1), feature_map_indices_grid)  # (B, 1, 3, FH*FW)

        cam_points = (uv.unsqueeze(3) * depths).view(batch_size, 1, 3, -1)  # (B, 1, 3, D*FH*FW)
        world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2).contiguous() \
            .view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

        preds["world_points"] = world_points

        num_world_points = world_points.size(-1)
        assert num_world_points == feature_height * feature_width * num_depth / 4

        point_features = self.feature_fetcher(feature_list, world_points, cam_intrinsic, cam_extrinsic)
        ref_feature = coarse_feature_maps[0]
        #print("before ref feature:", ref_feature.size())
        ref_feature = ref_feature[:, :, ::2,::2].contiguous()
        #print("after ref feature:", ref_feature.size())
        ref_feature = ref_feature.unsqueeze(2).expand(-1, -1, num_depth, -1, -1)\
                        .contiguous().view(batch_size,feature_channels,-1)
        point_features[:, 0, :, :] = ref_feature

        avg_point_features = torch.mean(point_features, dim=1)
        avg_point_features_2 = torch.mean(point_features ** 2, dim=1)

        point_features = avg_point_features_2 - (avg_point_features ** 2)

        cost_volume = point_features.view(batch_size, feature_channels, num_depth, feature_height // 2, feature_width // 2)

        # torch.cuda.synchronize()
        # timer_estimation.showTime('warping')

        filtered_cost_volume = self.coarse_vol_conv(cost_volume).squeeze(1)

        probability_volume = F.softmax(-filtered_cost_volume, dim=1)
        depth_volume = []
        for i in range(batch_size):
            depth_array = torch.linspace(depth_start[i], depth_end[i], num_depth, device=depth_start.device)
            depth_volume.append(depth_array)
        depth_volume = torch.stack(depth_volume, dim=0)  # (B, D)
        depth_volume = depth_volume.view(batch_size, num_depth, 1, 1).expand(probability_volume.shape)
        pred_depth_img = torch.sum(depth_volume * probability_volume, dim=1).unsqueeze(1)  # (B, 1, FH, FW)
        preds["raw_depth_map"] = pred_depth_img
        prob_map = get_propability_map(probability_volume, pred_depth_img, depth_start, depth_interval)
        # torch.cuda.synchronize()
        # timer_estimation.showTime('regression')

        # image guided depth map propagation
        pred_depth_img = F.interpolate(pred_depth_img, (feature_height, feature_width), mode="nearest")
        preds["interpolate_depth_map"] = pred_depth_img
        prob_map = F.interpolate(prob_map, (feature_height, feature_width), mode="bilinear")
        pred_depth_img = self.propagation_net(pred_depth_img, img_list[:, 0, :, :, :])

        preds["coarse_depth_map"] = pred_depth_img
        preds["coarse_prob_map"] = prob_map
        # torch.cuda.synchronize()
        # timer_estimation.showTime('propagation')
        # timer_estimation.summary()
        if isGN:
            feature_pyramids = {}
            chosen_conv = ["conv1", "conv2"]
            for conv in chosen_conv:
                feature_pyramids[conv] = []
            for i in range(num_view):
                curr_img = img_list[:, i, :, :, :]
                curr_feature_pyramid = self.flow_img_conv(curr_img)
                for conv in chosen_conv:
                    feature_pyramids[conv].append(curr_feature_pyramid[conv])

            for conv in chosen_conv:
                feature_pyramids[conv] = torch.stack(feature_pyramids[conv], dim=1)

            if isTest:
                for conv in chosen_conv:
                    feature_pyramids[conv] = torch.detach(feature_pyramids[conv])

            def get_grident(depth_map, all_features, uv, cam_intrinsic, cam_extrinsic, R_inv, t, batch_size):
                delta = 0.01
                interval_depth_map_l = depth_map - delta
                interval_depth_map_r = depth_map + delta
                cam_points_l = (uv * interval_depth_map_l.view(batch_size, 1, 1, -1))
                cam_points_r = (uv * interval_depth_map_r.view(batch_size, 1, 1, -1))
                world_points_l = torch.matmul(R_inv[:, 0:1, :, :], cam_points_l - t[:, 0:1, :, :]).transpose(1, 2) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)
                world_points_r = torch.matmul(R_inv[:, 0:1, :, :], cam_points_r - t[:, 0:1, :, :]).transpose(1, 2) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

                point_features_l = \
                    self.feature_fetcher(all_features, world_points_l, cam_intrinsic, cam_extrinsic)
                point_features_r = \
                    self.feature_fetcher(all_features, world_points_r, cam_intrinsic, cam_extrinsic)

                curr_grident = (point_features_r[:, 1:, ...] - point_features_l[:, 1:, ...]) / (2 * delta)

                return curr_grident.unsqueeze(4)

            def gn_update(estimated_depth_map, interval, image_scale, it, blending):
                nonlocal chosen_conv
                # print(estimated_depth_map.size(), image_scale)
                # torch.cuda.synchronize()
                # timer_refine = PICTimer.getTimer('refine')
                # timer_refine.startTimer()
                flow_height, flow_width = list(estimated_depth_map.size())[2:]
                if flow_height != int(img_height * image_scale):
                    flow_height = int(img_height * image_scale)
                    flow_width = int(img_width * image_scale)
                    estimated_depth_map = F.interpolate(estimated_depth_map, (flow_height, flow_width), mode="nearest")
                    # blending = F.interpolate(blending, (flow_height, flow_width), mode="nearest")
                else:
                    # if it is the same size return directly
                    #return estimated_depth_map
                    pass
                # blending = F.interpolate(blending, (int(img_height * image_scale), int(img_width * image_scale)), mode="nearest")
                if isTest:
                    estimated_depth_map = estimated_depth_map.detach()
                # torch.cuda.synchronize()
                # timer_refine.showTime('resize depth map')
                # GN step
                cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()
                if isTest:
                    cam_intrinsic[:, :, :2, :3] *= image_scale
                else:
                    cam_intrinsic[:, :, :2, :3] *= (4 * image_scale)

                ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
                feature_map_indices_grid = get_pixel_grids(flow_height, flow_width) \
                    .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device)

                uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1),
                                  feature_map_indices_grid)  # (B, 1, 3, FH*FW)

                interval_depth_map = estimated_depth_map
                cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
                world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)
                # torch.cuda.synchronize()
                # timer_refine.showTime('mapping')

                grad_pts = self.point_grad_fetcher(world_points, cam_intrinsic, cam_extrinsic)

                R_tar_ref = torch.bmm(R.view(batch_size * num_view, 3, 3),
                                      R_inv[:, 0:1, :, :].repeat(1, num_view, 1, 1).view(batch_size * num_view, 3, 3))

                R_tar_ref = R_tar_ref.view(batch_size, num_view, 3, 3)
                d_pts_d_d = uv.unsqueeze(-1).permute(0, 1, 3, 2, 4).contiguous().repeat(1, num_view, 1, 1, 1)
                d_pts_d_d = R_tar_ref.unsqueeze(2) @ d_pts_d_d
                d_uv_d_d = torch.bmm(grad_pts.view(-1, 2, 3), d_pts_d_d.view(-1, 3, 1)).view(batch_size, num_view, 1,
                                                                                             -1, 2, 1)
                # torch.cuda.synchronize()
                # timer_refine.showTime('d_d')

                all_features = []
                for conv in chosen_conv:
                    curr_feature = feature_pyramids[conv]
                    c, h, w = list(curr_feature.size())[2:]
                    curr_feature = curr_feature.contiguous().view(-1, c, h, w)
                    curr_feature = F.interpolate(curr_feature, (flow_height, flow_width), mode="bilinear")
                    curr_feature = curr_feature.contiguous().view(batch_size, num_view, c, flow_height, flow_width)

                    all_features.append(curr_feature)

                all_features = torch.cat(all_features, dim=2)
                # replace feature with rgb
                # all_features = F.interpolate(img_list.squeeze(0), (flow_height, flow_width), mode="bilinear").unsqueeze(0)
                # torch.cuda.synchronize()
                # timer_refine.showTime('feature resize')

                if isTest:
                    point_features, point_features_grad = \
                        self.feature_grad_fetcher.test_forward(all_features, world_points, cam_intrinsic, cam_extrinsic)
                else:
                    point_features, point_features_grad = \
                        self.feature_grad_fetcher(all_features, world_points, cam_intrinsic, cam_extrinsic)
                # torch.cuda.synchronize()
                # timer_refine.showTime('warping and d_uv')
                c = all_features.size(2)
                d_uv_d_d_tmp = d_uv_d_d.repeat(1, 1, c, 1, 1, 1)
                # print("d_uv_d_d tmp size:", d_uv_d_d_tmp.size())
                J = point_features_grad.view(-1, 1, 2) @ d_uv_d_d_tmp.view(-1, 2, 1)
                print(point_features_grad.view(-1, 1, 2).size(), d_uv_d_d_tmp.view(-1, 2, 1).size())
                J = J.view(batch_size, num_view, c, -1, 1)[:, 1:, ...].contiguous()
                ## 48*4 -> 48*4, blending
                # blending = blending.view(1, 4, 1, -1).unsqueeze(4)
                # J = J * blending
                J = J.permute(0, 3, 1, 2, 4).contiguous().view(-1, c * (num_view - 1), 1)
                ## 48*4 -> 48 by mean
                # J = torch.mean(J.permute(0, 3, 2, 1, 4).contiguous(), dim=3).squeeze(3).permute(1, 2, 0)
                ## 48*4 -> 48*4, but scale to 0.25
                #J = J * 0.25
                # print(J.size())

                ### Finite element difference ###
                # J = get_grident(estimated_depth_map, all_features, uv, cam_intrinsic, cam_extrinsic, R_inv, t,
                #                 batch_size)
                # J = J.contiguous().permute(0, 3, 1, 2, 4).contiguous().view(-1, c * (num_view - 1), 1)
                # torch.cuda.synchronize()
                # timer_refine.showTime('compute jacobi')

                resid = point_features[:, 1:, ...] - point_features[:, 0:1, ...]
                first_resid = torch.sum(torch.abs(resid), dim=(1, 2))
                # print(resid.size())
                ## 48*4 -> 48*4, blending
                # resid = resid * blending.squeeze(4)
                resid = resid.permute(0, 3, 1, 2).contiguous().view(-1, c * (num_view - 1), 1)
                ## 48*4 -> 48 by mean
                # resid = torch.mean(resid.permute(0, 3, 2, 1).contiguous(), dim=3).permute(1, 2, 0)
                ## 48*4 -> 48*4, but scale to 0.25
                #resid = resid * 0.25

                # torch.cuda.synchronize()
                # timer_refine.showTime('compute loss')

                J_t = torch.transpose(J, 1, 2)
                H = J_t @ J
                b = -J_t @ resid
                delta = b / (H + 1e-6)
                # torch.cuda.synchronize()
                # timer_refine.showTime('compute delta')
                # #print(delta.size())
                _, _, h, w = estimated_depth_map.size()

                
                # ####vis grad_map####
                # grad_map = delta.view(-1, 1, h, w)[0][0]
                # grad_map = grad_map.cpu().numpy()
                # gmax, gmin = 0.13, -0.5#np.max(grad_map), np.min(grad_map)
                # grad_map = (grad_map - gmin) / (gmax - gmin) * 255.0
                # grad_map = grad_map.astype('uint8')
                # cv2.imshow("grad", grad_map)
                # cv2.waitKey()
                
                flow_result = estimated_depth_map + delta.view(-1, 1, h, w)
                # torch.cuda.synchronize()
                # timer_refine.showTime('update depth map')

                # check update results
                interval_depth_map = flow_result
                cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
                world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

                point_features = \
                    self.feature_fetcher(all_features, world_points, cam_intrinsic, cam_extrinsic)

                resid = point_features[:, 1:, ...] - point_features[:, 0:1, ...]
                second_resid = torch.sum(torch.abs(resid), dim=(1, 2))
                # print(first_resid.size(), second_resid.size())
                # torch.cuda.synchronize()
                # timer_refine.showTime('compute new loss')

                # only accept good update
                flow_result = torch.where((second_resid < first_resid).view(batch_size, 1, flow_height, flow_width),
                                          flow_result, estimated_depth_map)
                # torch.cuda.synchronize()
                # timer_refine.showTime('accept good update')
                # timer_refine.summary()
                return flow_result

            for i, (img_scale, inter_scale) in enumerate(zip(img_scales, inter_scales)):
                if isTest:
                    pred_depth_img = torch.detach(pred_depth_img)
                    print("update: {}".format(i))
                torch.cuda.synchronize()
                begin_time = time.time()
                flow = gn_update(pred_depth_img, inter_scale* depth_interval, img_scale, i, blending)
                torch.cuda.synchronize()
                print('scale:{}, time:{}\n'.format(img_scale, time.time() - begin_time))
                preds["flow{}".format(i+1)] = flow
                pred_depth_img = flow

        return preds


class PointMVSNetLoss(nn.Module):
    def __init__(self, valid_threshold):
        super(PointMVSNetLoss, self).__init__()
        self.maeloss = MAELoss()
        self.valid_maeloss = Valid_MAELoss(valid_threshold)

    def forward(self, preds, labels, isFlow):
        gt_depth_img = labels["gt_depth_img"]
        depth_interval = labels["cam_params_list"][:, 0, 1, 3, 1]

        coarse_depth_map = preds["coarse_depth_map"]
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
        coarse_loss = self.maeloss(coarse_depth_map, resize_gt_depth, depth_interval)

        losses = {}
        losses["coarse_loss"] = coarse_loss

        if isFlow:
            flow1 = preds["flow1"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))
            flow1_loss = self.maeloss(flow1, resize_gt_depth, 0.75 * depth_interval)
            losses["flow1_loss"] = flow1_loss

            flow2 = preds["flow2"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))
            flow2_loss = self.maeloss(flow2, resize_gt_depth, 0.375 * depth_interval)
            losses["flow2_loss"] = flow2_loss

        for k in losses.keys():
            losses[k] /= float(len(losses.keys()))

        return losses


def cal_less_percentage(pred_depth, gt_depth, depth_interval, threshold):
    shape = list(pred_depth.size())
    mask_valid = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    denom = torch.sum(mask_valid) + 1e-7
    interval_image = depth_interval.view(-1, 1, 1, 1).expand(shape)
    abs_diff_image = torch.abs(pred_depth - gt_depth) / interval_image

    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct


def cal_valid_less_percentage(pred_depth, gt_depth, before_depth, depth_interval, threshold, valid_threshold):
    shape = list(pred_depth.size())
    mask_true = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    interval_image = depth_interval.view(-1, 1, 1, 1).expand(shape)
    abs_diff_image = torch.abs(pred_depth - gt_depth) / interval_image

    if before_depth.size(2) != shape[2]:
        before_depth = F.interpolate(before_depth, (shape[2], shape[3]))

    diff = torch.abs(before_depth - gt_depth) / interval_image
    mask_valid = (diff < valid_threshold).type(torch.float)
    mask_valid = mask_valid * mask_true

    denom = torch.sum(mask_valid) + 1e-7
    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct


class PointMVSNetMetric(nn.Module):
    def __init__(self, valid_threshold):
        super(PointMVSNetMetric, self).__init__()
        self.valid_threshold = valid_threshold

    def forward(self, preds, labels, isFlow):
        gt_depth_img = labels["gt_depth_img"]
        depth_interval = labels["cam_params_list"][:, 0, 1, 3, 1]

        coarse_depth_map = preds["coarse_depth_map"]
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))

        less_one_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 1.0)
        less_three_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 3.0)

        metrics = {
            "<1_pct_cor": less_one_pct_coarse,
            "<3_pct_cor": less_three_pct_coarse,
        }

        if isFlow:
            flow1 = preds["flow1"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))

            less_one_pct_flow1 = cal_valid_less_percentage(flow1, resize_gt_depth, coarse_depth_map,
                                                           0.75 * depth_interval, 1.0, self.valid_threshold)
            less_three_pct_flow1 = cal_valid_less_percentage(flow1, resize_gt_depth, coarse_depth_map,
                                                             0.75 * depth_interval, 3.0, self.valid_threshold)

            metrics["<1_pct_flow1"] = less_one_pct_flow1
            metrics["<3_pct_flow1"] = less_three_pct_flow1

            flow2 = preds["flow2"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))

            less_one_pct_flow2 = cal_valid_less_percentage(flow2, resize_gt_depth, flow1,
                                                           0.375 * depth_interval, 1.0, self.valid_threshold)
            less_three_pct_flow2 = cal_valid_less_percentage(flow2, resize_gt_depth, flow1,
                                                             0.375 * depth_interval, 3.0, self.valid_threshold)

            metrics["<1_pct_flow2"] = less_one_pct_flow2
            metrics["<3_pct_flow2"] = less_three_pct_flow2

        return metrics


def build_pointmvsnet(cfg):
    net = FastMVSNet(
        img_base_channels=cfg.MODEL.IMG_BASE_CHANNELS,
        vol_base_channels=cfg.MODEL.VOL_BASE_CHANNELS,
        flow_channels=cfg.MODEL.FLOW_CHANNELS,
    )

    loss_fn = PointMVSNetLoss(
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
    )

    metric_fn = PointMVSNetMetric(
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
    )

    return net, loss_fn, metric_fn


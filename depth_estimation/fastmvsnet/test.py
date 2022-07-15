#!/usr/bin/env python
import argparse
import os.path as osp
import logging
import time
import sys

import cv2
from cv2 import INTER_LINEAR
import numpy

sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn

from fastmvsnet.config import load_cfg_from_file
from fastmvsnet.utils.io import mkdir
from fastmvsnet.utils.logger import setup_logger
from fastmvsnet.model import build_pointmvsnet as build_model
from fastmvsnet.utils.checkpoint import Checkpointer
from fastmvsnet.dataset import build_data_loader
from fastmvsnet.utils.metric_logger import MetricLogger
from fastmvsnet.utils.eval_file_logger import eval_file_logger


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Fast-MVSNet Evaluation")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="../configs/dtu.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--cpu",
        action='store_true',
        default=False,
        help="whether to only use cpu for test",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def test_model(model,
               image_scales,
               inter_scales,
               data_loader,
               folder,
               isCPU=False,
               ):
    logger = logging.getLogger("fastmvsnet.train")
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()
    path_list = []
    # blending_weights = numpy.load('/home/wjk/workspace/PyProject/FastMVSNet/tools/image/blending_weights.npy', allow_pickle=True)
    # blending_weights = numpy.sqrt(blending_weights)
    # blending_weights = numpy.ones_like(blending_weights)
    # blending_weights = numpy.random.random([160, 96, 4, 5])
    # blending_weights[blending_weights < 0.5] = 0
    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end
            # curr_ref_img_path = data_batch["ref_img_path"][0]
            # path_list.extend(curr_ref_img_path)
            if not isCPU:
                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}

            weight = None#torch.tensor(blending_weights[:, :, :, iteration], dtype=torch.float32, device='cuda:0').permute(2, 0, 1).unsqueeze(0)

            preds = model(data_batch=data_batch, img_scales=image_scales, inter_scales=inter_scales, blending=None, isGN=False, isTest=True)
            
            for key in preds.keys():
                
                #if key == 'world_points':
                #     continue
                # if key != 'flow2':
                #     continue
                if key!= 'coarse_depth_map': continue
                tmp = preds[key][0,0].cpu().numpy()
                #tmp = 1.0 / tmp
                maxn = numpy.max(tmp)#1.6###1.6
                minn = numpy.min(tmp)#0.7###1.0
                print('max:{},min:{}'.format(maxn, minn))
                tmp = (tmp - minn) / (maxn - minn) * 255.0
                # tmp = tmp.astype('uint8')
                # tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_RAINBOW)
                
                # print(f"{key} shape: ", tmp.shape, tmp.dtype)
                # cv2.imwrite(f"/home/wph/FastMVSNet/wph_test/{key}_{iteration}_preds{tmp.shape}_of_{tmp.dtype}.jpg", tmp)
                
                if key == "coarse_depth_map":
                    tmp_ori = cv2.resize(tmp,dsize=None, fx=4, fy=4, interpolation=INTER_LINEAR)
                    cv2.imwrite(f"/home/wph/pipe_transmission/depth_estimation/FastMVSNet/wph_test/0715_{key}_{iteration}_preds.jpg", tmp_ori)
                
                # cv2.imshow(key, tmp)
                # cv2.waitKey()
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            # logger.info(
            #     "{} finished.".format(curr_ref_img_path) + str(meters))
            # eval_file_logger(data_batch, preds, curr_ref_img_path, folder)


def test(cfg, output_dir, isCPU=False):
    print("!test output_dir", output_dir)
    logger = logging.getLogger("fastmvsnet.tester")
    # build model
    model, _, _ = build_model(cfg)
    if not isCPU:
        model = nn.DataParallel(model).cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir)
    # # 0714 问题就在于checkpointer
    # if cfg.TEST.WEIGHT:
    #     weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
    #     print(f"! weight_path: {weight_path}")
    #     checkpointer.load(weight_path, resume=False)
    # else:
    #     checkpointer.load(None, resume=True)
    # print(f"! Model: {model}")
    stat_dict = torch.load(cfg.TEST.WEIGHT, map_location=torch.device("cpu"))
    model.load_state_dict(stat_dict.pop("model"), strict = False)

    # build data loader
    test_data_loader = build_data_loader(cfg, mode="test")
    start_time = time.time()
    test_model(model,
               image_scales=cfg.MODEL.TEST.IMG_SCALES,
               inter_scales=cfg.MODEL.TEST.INTER_SCALES,
               data_loader=test_data_loader,
               folder=output_dir.split("/")[-1],
               isCPU=isCPU,
               )
    test_time = time.time() - start_time
    logger.info("Test forward time: {:.2f}s".format(test_time))


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    cfg = load_cfg_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    assert cfg.TEST.BATCH_SIZE == 1

    isCPU = args.cpu

    output_dir = cfg.OUTPUT_DIR
    print("! :",output_dir)
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        mkdir(output_dir)

    # print(f"wph: {output_dir}")
    logger = setup_logger("fastmvsnet", output_dir, prefix="test")
    if isCPU:
        logger.info("Using CPU")
    else:
        logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))
    
    output_dir = "single_frame"
    test(cfg, output_dir, isCPU=isCPU)


if __name__ == "__main__":
    main()

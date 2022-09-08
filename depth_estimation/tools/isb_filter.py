"""
=======================================================================
General Information
-------------------
This is a GPU-based python implementation of the following paper:
Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
Proc. IEEE Computer Vision and Pattern Recognition (CVPR 2021, Oral)
Visit our project http://vclab.kaist.ac.kr/cvpr2021p1/ for more details.

Please cite this paper if you use this code in an academic publication.
Bibtex: 
@InProceedings{Meuleman_2021_CVPR,
    author = {Andreas Meuleman and Hyeonjoong Jang and Daniel S. Jeon and Min H. Kim},
    title = {Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images},
    booktitle = {CVPR},
    month = {June},
    year = {2021}
}
==========================================================================
License Information
-------------------
CC BY-NC-SA 3.0
Andreas Meuleman and Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:
Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.
The use of the software is for Non-Commercial Purposes only. As used in this Agreement, “Non-Commercial Purpose” means for the purpose of education or research in a non-commercial organisation only. “Non-Commercial Purpose” excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].
Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
Please refer to license.txt for more details.
=======================================================================
"""
import torch
import cupy
import math
import cv2
import numpy as np
from tools.utils import create_dir

import os

class ISB_Filter:
    def __init__(self, candidate_count, resolution, device, path, SR_resolution = -1):
        """
        Fast edge-preserving filter. 
        Compile CUDA functions and allocate intermediate scales
        Args:
            candidate_count: number of channels of the quantity to be filter.
                Corresponds to the number of distance candidates for cost volumes.
            resolution: size of the image (cols, rows) (guide and cost should match)
            device: CUDA-enabled GPU used for processing
        """
        # Read and compile CUDA functions
        with open('depth_estimation/tools/vec_utils.cuh', 'r',encoding='UTF-8') as f:
            utils_source = f.read()
        with open('depth_estimation/tools/isb_filter.cu', 'r',encoding='UTF-8') as f:
            cuda_source = utils_source + f.read()
        cuda_source = cuda_source.replace("CANDIDATE_COUNT", str(candidate_count))
        module = cupy.RawModule(code=cuda_source)

        self.device = device
        self.dataset_path = path

        self.guide_downsample_cuda = module.get_function("guideDownsample2xKernel")
        self.guide_upsample_cuda = module.get_function("guideUpsample2xKernel")

        # Allocate intermediate scales
        cols = resolution[0]
        rows = resolution[1]
        if candidate_count == 1:
            if SR_resolution == -1:
                self.scale_count = int(min(math.log2(cols), math.log2(rows)) - 1) - 3 #- 3 # control the propagation area
            else:
                self.scale_count = int(math.log2(SR_resolution[0] / resolution[0])) + 1
                cols = SR_resolution[0]
                rows = SR_resolution[1]
        else:
            self.scale_count = int(min(math.log2(cols), math.log2(rows)) - 1) -1#- 3

        self.guides = []
        self.costs = []
        for scale in range(0, self.scale_count):
            self.guides.append( 
                torch.zeros([math.ceil(rows/(2**scale)), math.ceil(cols/(2**scale)), 3], 
                            dtype=torch.uint8, device=self.device))
            
            self.costs.append(
                torch.zeros([candidate_count, math.ceil(rows/(2**scale)), math.ceil(cols/(2**scale))], device=self.device))

    def apply(self, guide, cost, sigma_i, sigma_s, index = 0):
        """
        Apply the filter to a cost volume (or another quantity to be smoothed)
        Args:
            guide: [rows, cols, 3] Guide for edge-preserving filtering (uint8).
                using YUV or yCbCr colour spaces usually improves guidance
            cost: [candidate_count, rows, cols] Cost volume to be aggregated (float32)
            sigma_i: Edge preservation parameter. Lower values preserve edges during cost volume filtering
            sigma_s: Smoothing parameter. Higher values give more weight to coarser scales during filtering
        Returns:
            guide: [rows, cols, 3] Filtered guide
            cost: [candidate_count, rows, cols] Filtered cost volume
        """
        self.guides[0] = guide
        self.costs[0] = cost
        var_inv_s = 1 / (2 * sigma_s * sigma_s)
        var_inv_i = 1 / (2 * sigma_i * sigma_i)
        for scale in range(1, self.scale_count):
            block_size = min(256, 2**math.ceil(math.log2(self.guides[scale].shape[0] * self.guides[scale].shape[1])))
            grid_size = math.ceil(self.guides[scale].shape[0] * self.guides[scale].shape[1] / block_size)
            self.guide_downsample_cuda(
                block=(block_size,),
                grid=(grid_size,),
                args=(
                    self.guides[scale - 1].data_ptr(),
                    self.costs[scale - 1].data_ptr(),
                    self.guides[scale - 1].shape[0],
                    self.guides[scale - 1].shape[1],
                    self.guides[scale].data_ptr(),
                    self.costs[scale].data_ptr(),
                    self.guides[scale].shape[0],
                    self.guides[scale].shape[1],
                    cupy.float32(var_inv_i)
                )
            )

        # origin = guide.cpu().numpy()
        # origin = cv2.cvtColor(origin, cv2.COLOR_YCrCb2BGR)
        # for guide_color in self.guides:
        #     tmp = cv2.cvtColor(guide_color.cpu().numpy(), cv2.COLOR_YCrCb2BGR)
        #     cv2.imshow("guide", tmp)
        #     cv2.waitKey()
        #     cv2.imshow("guide", origin)
        #     cv2.waitKey()
        #     origin = cv2.resize(origin, tuple([int(math.ceil(origin.shape[1]/2)),int(math.ceil(origin.shape[0]/2))]), cv2.INTER_LANCZOS4)

        # if self.costs[0].shape[0] == 1:
        #     for cost in self.costs:
        #         dis_map = cost[0].cpu().numpy()
        #         dis_map[dis_map <= 0] = 100
        #         dis_map = 1.0 / dis_map
        #         maxs = np.max(dis_map)
        #         mins = np.min(dis_map)
        #         dis_map = 255.0 * (dis_map - mins) / (maxs - mins)
        #         dis_map = dis_map.astype('uint8')
        #         # output_path = self.dataset_path + '//output//filter//'
        #         # create_dir(output_path)
        #         # cv2.imwrite(output_path + 'cam(' + str(index) + ')down' + str(dis_map.shape) + '.png', dis_map)
        #         cv2.imshow("depth", dis_map)
        #         cv2.waitKey()

        for scale in range(self.scale_count - 2, -1, -1):
            distance = 2**scale - 0.5
            weight_down = (math.exp(-(distance * distance) * var_inv_s))
            weight_up = 1 - weight_down

            block_size = min(256, 2**math.ceil(
                math.log2(self.guides[scale + 1].shape[0] * self.guides[scale + 1].shape[1])))
            grid_size = math.ceil(self.guides[scale + 1].shape[0] * self.guides[scale + 1].shape[1] / block_size)

            self.guide_upsample_cuda(
                block=(block_size,),
                grid=(grid_size,),
                args=(
                    self.guides[scale + 1].data_ptr(),
                    self.costs[scale + 1].data_ptr(),
                    self.guides[scale + 1].shape[0],
                    self.guides[scale + 1].shape[1],
                    self.guides[scale].data_ptr(),
                    self.costs[scale].data_ptr(),
                    self.guides[scale].shape[0],
                    self.guides[scale].shape[1],
                    cupy.float32(weight_up),
                    cupy.float32(weight_down),
                    cupy.float32(var_inv_i)
                )
            )

        # for i in range(len(self.guides)-1,-1,-1):
        #     tmp = cv2.cvtColor(self.guides[i].cpu().numpy(), cv2.COLOR_YCrCb2BGR)
        #     cv2.imshow("guide", tmp)
        #     cv2.waitKey()

        # if self.costs[0].shape[0] == 1:
        #     for i in range(len(self.guides) - 1, -1, -1):
        #         dis_map = self.costs[i][0].cpu().numpy()
        #         dis_map[dis_map <= 0] = 100
        #         dis_map = 1.0 / dis_map
        #         maxs = np.max(dis_map)
        #         mins = np.min(dis_map)
        #         dis_map = 255.0 * (dis_map - mins) / (maxs - mins)
        #         dis_map = dis_map.astype('uint8')
        #         # cv2.imshow("depth", dis_map)
        #         # cv2.waitKey()
        #         output_path = self.dataset_path + '//output//filter//'
        #         create_dir(output_path)
        #         cv2.imwrite(output_path + 'cam(' + str(index) + ')up' + str(dis_map.shape) + '.png', dis_map)

        return self.costs[0], self.guides
        #return self.costs[0], self.guides[0]

    def SR(self, SR_guide, guide, cost, sigma_i, sigma_s, index = 0):
        self.guides[0] = SR_guide
        self.guides[-1] = guide
        self.costs[-1] = cost
        var_inv_s = 1 / (2 * sigma_s * sigma_s)
        var_inv_i = 1 / (2 * sigma_i * sigma_i)
        for scale in range(1, self.scale_count - 1):
            block_size = min(256, 2 ** math.ceil(math.log2(self.guides[scale].shape[0] * self.guides[scale].shape[1])))
            grid_size = math.ceil(self.guides[scale].shape[0] * self.guides[scale].shape[1] / block_size)
            self.guide_downsample_cuda(
                block=(block_size,),
                grid=(grid_size,),
                args=(
                    self.guides[scale - 1].data_ptr(),
                    self.costs[scale - 1].data_ptr(),
                    self.guides[scale - 1].shape[0],
                    self.guides[scale - 1].shape[1],
                    self.guides[scale].data_ptr(),
                    self.costs[scale].data_ptr(),
                    self.guides[scale].shape[0],
                    self.guides[scale].shape[1],
                    cupy.float32(var_inv_i)
                )
            )

        # for guide_color in self.guides:
        #     tmp = cv2.cvtColor(guide_color.cpu().numpy(), cv2.COLOR_YCrCb2BGR)
        #     cv2.imshow("guide", tmp)
        #     cv2.waitKey()

        # super_scale = 3
        # super_guides = [self.guides[0]]
        # super_costs = [self.costs[0]]
        # for scale in range(super_scale):
        #     prev_shape = super_guides[0].shape
        #     super_guides.insert(0, torch.zeros([prev_shape[0] * 2, prev_shape[1] * 2, 3], dtype=torch.uint8, device=self.device))
        #     super_costs.insert(0, -torch.ones([1, prev_shape[0] * 2, prev_shape[1] * 2], device=self.device))

        for scale in range(self.scale_count - 2, -1, -1):
            distance = 2 ** scale - 0.5
            #weight_down = (math.exp(-(distance * distance) * var_inv_s))
            weight_down  = 1
            weight_up = 1 - weight_down

            block_size = min(256, 2 ** math.ceil(
                math.log2(self.guides[scale + 1].shape[0] * self.guides[scale + 1].shape[1])))
            grid_size = math.ceil(self.guides[scale + 1].shape[0] * self.guides[scale + 1].shape[1] / block_size)

            self.guide_upsample_cuda(
                block=(block_size,),
                grid=(grid_size,),
                args=(
                    self.guides[scale + 1].data_ptr(),
                    self.costs[scale + 1].data_ptr(),
                    self.guides[scale + 1].shape[0],
                    self.guides[scale + 1].shape[1],
                    self.guides[scale].data_ptr(),
                    self.costs[scale].data_ptr(),
                    self.guides[scale].shape[0],
                    self.guides[scale].shape[1],
                    cupy.float32(weight_up),
                    cupy.float32(weight_down),
                    cupy.float32(var_inv_i)
                )
            )

        # for i in range(len(self.guides) - 1, -1, -1):
        #     dis_map = self.costs[i][0].cpu().numpy()
        #     dis_map[dis_map <= 0] = 100
        #     dis_map = 1.0 / dis_map
        #     maxs = np.max(dis_map)
        #     mins = np.min(dis_map)
        #     dis_map = 255.0 * (dis_map - mins) / (maxs - mins)
        #     dis_map = dis_map.astype('uint8')
        #     # cv2.imshow("depth", dis_map)
        #     # cv2.waitKey()
        #     output_path = self.dataset_path + '//output//filter//'
        #     create_dir(output_path)
        #     cv2.imwrite(output_path + 'cam(' + str(index) + ')SR' + str(dis_map.shape) + '.png', dis_map)
        return self.costs[0], self.guides
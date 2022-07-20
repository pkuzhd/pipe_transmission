import torch
import numpy as np

from fastmvsnet.utils.preprocess import  norm_image,  crop_dtu_input, scale_dtu_input

def load_cam_paras(file, num_depth=0, interval_scale=1.0):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = num_depth
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

def write_cam_dtu(file, cam):
    # f = open(file, "w")
    f = open(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write(
        '\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

def get_ori_cam_paras(num_view, paths, num_virtual_plane = 128, interval_scale = 1.6):
    # get images and cams' paras
    cams = []
    for view in range(num_view):
        cam = load_cam_paras(open(paths[view]),
                                num_depth=num_virtual_plane,
                                interval_scale=interval_scale)

        cams.append(cam)

    return cams
    
def adjust_cam_para(cams, crops):
    num_view = len(cams)
    ad_cams = cams.copy()
    for i in range(num_view):
        ad_cams[i][1][0][2] -= crops[i][2]
        ad_cams[i][1][1][2] -= crops[i][3]
    
    return ad_cams


def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam

def build_data_forFast_sc(Imgs, Cams, alpha, device, height, width):
    imgs = Imgs.copy()
    cams = Cams.copy()

    h_scale = float(height) / imgs[0].shape[0]
    w_scale = float(width) / imgs[0].shape[1]
    # print(imgs[0].shape)
    if h_scale > 1 or w_scale > 1:
        print("max_h, max_w should < W and H!")
        exit()
    resize_scale = h_scale
    if w_scale > h_scale:
        resize_scale = w_scale
    
    # print("! build img type: ", imgs[0].dtype, imgs[0].shape)


    scaled_input_images, scaled_input_cams = scale_dtu_input(imgs, cams, depth_image=None, scale=resize_scale)
    # print("! build scale img type: ", scaled_input_images[0].dtype, scaled_input_images[0].shape)

    # crop to fit network

    croped_images, croped_cams = crop_dtu_input(scaled_input_images, scaled_input_cams,height=height, width=width, base_image_size=64, depth_image=None)

    for i, image in enumerate(croped_images):
        croped_images[i] = norm_image(image)

    
    img_list = np.stack(croped_images, axis=0)
    cam_params_list = np.stack(croped_cams, axis=0)
    # print(f"! cam_params_list: {cam_params_list.shape}")

    img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
    cam_params_list = torch.tensor(cam_params_list).float()

    img_list = img_list.unsqueeze(0).to(device)
    cam_params_list = cam_params_list.unsqueeze(0).to(device)

    # print(f"! build img_list shape {img_list.shape}")
    # print(f"! build cam_params_list shape {cam_params_list.shape}")

    return img_list, cam_params_list

def build_data_forFast(Imgs, Cams, alpha, device):

    imgs = Imgs.copy()
    cams = Cams.copy()

    # for i, image in enumerate(imgs):
    #     imgs[i] = norm_image(image)
    for i, image in enumerate(imgs):
        imgs[i] = image / 255.0

    img_list = np.stack(imgs, axis=0)
    cam_params_list = np.stack(cams, axis=0)
    # print(f"! cam_params_list: {cam_params_list.shape}")
    # cam_pos_list = np.stack(camspos, axis=0)

    img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
    cam_params_list = torch.tensor(cam_params_list).float()

    img_tensor = img_list.unsqueeze(0).to(device)
    cam_params_tensor = cam_params_list.unsqueeze(0).to(device)

    # img_tensor = img_tensor * alpha

    # print(f"! build img_list shape {img_list.shape}")
    # print(f"! build cam_params_list shape {cam_params_list.shape}")

    return img_tensor, cam_params_tensor
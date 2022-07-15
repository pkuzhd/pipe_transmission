import cv2
import torch
import torch.optim as optim
from os.path import join as jpath
import time

from IphoneUtils import readCameraInfo, generatePointCloud, savePointCloud, readCameraAndImage, myNormalize

from picutils import MyPerspectiveCamera, PICTimer

if __name__ == '__main__':
    frameIds = [_ for _ in range(0, 260, 100)]
    refCamIdx = 0
    srcCamIdx = -1

    device = 'cpu'
    device = 'cuda:1'

    cams, imgs, rawImgs, deps = readCameraAndImage(
        jpath("datasets", "iphone"), 
        jpath("datasets", "iphone", "imgs"), 
        jpath("datasets", "iphone", "depth"), 
        frameIds, device, getRaw=True
    )
    
    refCam, refImg, refRawImg, refDep = cams[refCamIdx], imgs[refCamIdx], rawImgs[refCamIdx], deps[refCamIdx]
    srcCam, srcImg, srcRawImg, srcDep = cams[srcCamIdx], imgs[srcCamIdx], rawImgs[srcCamIdx], deps[srcCamIdx]

    refRawImg = refRawImg / 255
    srcRawImg = srcRawImg / 255

    refDepHighRes = torch.nn.functional.interpolate(refDep.unsqueeze(0).unsqueeze(0), refRawImg.shape[:2]).squeeze(0)
    refDepHighRes.requires_grad = True

    rawRefCam = MyPerspectiveCamera.buildFromCamera(refCam, resize=refRawImg.shape[:2], device=device)
    rawSrcCam = MyPerspectiveCamera.buildFromCamera(srcCam, resize=srcRawImg.shape[:2], device=device)

    refDepHighRes = torch.nn.functional.interpolate(refDep.unsqueeze(0).unsqueeze(0), refRawImg.shape[:2]).squeeze(0)
    refDepHighRes = refDepHighRes.float()
    refDepHighRes.requires_grad = True

    optimizer = optim.SGD([refDepHighRes], lr=5, momentum=0.9)


    U_dst = torch.cat([rawRefCam.uv_grid, torch.ones(1, rawRefCam.imgH, rawRefCam.imgW, dtype=rawRefCam.uv_grid.dtype, device=rawRefCam.uv_grid.device)], dim=0).reshape(3, -1)
    basePoint, direction, _, _ = rawRefCam.uv2WorldLine(U_dst)
    basePoint = basePoint.float()
    direction = direction.float()
    normalize_base = torch.tensor([rawSrcCam.imgW * 0.5, rawSrcCam.imgH * 0.5], dtype=rawRefCam.uv_grid.dtype, device=rawRefCam.uv_grid.device).unsqueeze(1)
    eps = 1e-8
    mode='bilinear'
    # mode='nearest'
    padding_mode='zeros'
    align_corners=False
    posture = rawSrcCam.posture.float()
    k = rawSrcCam.k.float()
    with PICTimer.getTimer() as t:
        for _ in range(1_0):
            startT = time.time()

            optimizer.zero_grad()
            torch.cuda.synchronize()
            t.showTime("zero")
            tZero = time.time()

            #  _img_ref_hat = rawRefCam.fromCam(srcCam, refDepHighRes)
            # torch.cuda.synchronize()
            # t.showTime("warp")

            # img_ref_hat = _img_ref_hat(srcImg / 255)
            # torch.cuda.synchronize()
            # t.showTime("zero")
            # tZero = time.time()

            U_dst2XYWorld4 = basePoint + refDepHighRes.reshape(1, -1) * direction
            # grid = rawSrcCam.world2uv(U_dst2XYWorld4, eps)

            xYCameraSrc4 = posture @ U_dst2XYWorld4
            xYCameraSrc3 = xYCameraSrc4[:3]
            u_src3 = k @ xYCameraSrc3

            d_src = u_src3[2,:]
            u_src3 = u_src3 / (d_src.unsqueeze(0) + eps)

            grid = u_src3[:2]

            grid = (grid - normalize_base) / normalize_base
            grid = grid.reshape(2, rawRefCam.imgH, rawRefCam.imgW)
            grid = grid.permute(1, 2, 0)

            img_ref_hat = torch.nn.functional.grid_sample(srcRawImg.permute(2, 0, 1).unsqueeze(0), grid.unsqueeze(0), mode, padding_mode, align_corners).squeeze(0).permute(1, 2, 0)
            torch.cuda.synchronize()
            t.showTime("warp")
            tWarp = time.time()
            
            loss = ((refRawImg.unsqueeze(0) - img_ref_hat)**2).mean()
            torch.cuda.synchronize()
            t.showTime("loss")
            tLoss = time.time()
            
            
            loss.backward()
            torch.cuda.synchronize()
            t.showTime("back")
            tBack = time.time()

            optimizer.step()
            torch.cuda.synchronize()
            t.showTime("step")
            tStep = time.time()

            names = ['step', 'back', 'loss', 'warp', 'zero']
            times = [tStep - tBack, tBack - tLoss, tLoss - tWarp, tWarp - tZero, tZero - startT]

            print(["{}: {:.5f}ms".format(n, t*1000) for n, t in zip(names, times)])

            # continue
            # if _ % 1000 == 0:
            #     print(loss.data)
            #     torch.cuda.synchronize()
            #     print((time.time() - startT) * 1000 / 1000)
            #     startT = time.time()

            # tmpGrad = refDepHighRes.grad.squeeze(0).cpu().numpy()
            # tmpGrad = myNormalize(tmpGrad) * 255
            # cv2.imwrite('outputs/grad.jpg', tmpGrad)

            # refDepHighRes = refDepHighRes.detach().squeeze(0).cpu().numpy()
            # refDepHighRes = myNormalize(refDepHighRes) * 255
            # cv2.imwrite('outputs/grad_dep.jpg', refDepHighRes)

            # cv2.imwrite('outputs/grad_img.jpg', refRawImg.cpu().numpy())
            # cv2.imwrite('outputs/grad_hat.jpg', img_ref_hat.detach().squeeze(0).cpu().numpy()*255)

            # break

            t.forceShowTime()

    print(1)
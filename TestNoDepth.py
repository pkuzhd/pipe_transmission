import numpy as np
import time
import os

from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender

recevier = ImageReceiver()
recevier.open("./pipe_dir/pipe1")
print("pipe1 open")

sender = RGBDSender()
sender.open("./pipe_dir/pipe2")
print("pipe2 open")

dirs = './test_dir/'
if not os.path.exists(dirs):
    os.makedirs(dirs)

time_sum = [0, 0, 0, 0]
test_num = 400

for j in range(test_num):
    t1 = time.time()
    # data = ImageData(5, imgs)
    data = recevier.getData()
    t2 = time.time()
    rgbd_data_2 = RGBDData(
        data.N,
        data.imgs,
        [np.ones((800, 800), dtype=np.float32) * 0.5 for i in range(data.N)],
        [np.ones((800, 800), dtype=np.uint8) for i in range(data.N)],
        [(800, 800, 400, 200) for i in range(data.N)],
    )
    t3 = time.time()
    print(j, rgbd_data_2.crops)
    bytes = sender.sendData(rgbd_data_2)

    if (bytes == -1):
        print("pipe has been closed.")
        sender.close()
        break

    t4 = time.time()
    print(
        f"get time: {t2 - t1:.3f}, depth estimation time: {t3 - t2:.3f}, send time: {t4 - t3}, total time: {t4 - t1:.3f}")
    time_sum[0] += t2 - t1
    time_sum[1] += t3 - t2
    time_sum[2] += t4 - t3
    time_sum[3] += t4 - t1
    # for key in rgbd_data.keys():
    #     if key == 'crops' or key == 'num_view': continue
    #     print(f"rgbd-{key}: {rgbd_data[key][0].shape} {rgbd_data[key][0].dtype}")
    #
    # for key in rgbd_data.keys():
    #     if key == 'imgs':
    #         i = 0
    #         for img in rgbd_data[key]:
    #             cv2.imwrite(dirs + f"imgs_{i + 1}-{j + 1}.jpg", (img))
    #             i += 1
    #     elif key == 'depths':
    #         i = 0
    #         for depth in rgbd_data[key]:
    #
    #             maxn = 1.6#1.6###numpy.max(tmp)
    #             minn = 1.0#0.7###numpy.min(tmp)
    #             tmp = (depth - minn) / (maxn - minn) * 255.0
    #             tmp = tmp.astype('uint8')
    #             tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_RAINBOW)
    #             cv2.imwrite(dirs + f"depth_{i + 1}-{j + 1}.jpg", tmp)
    #             i += 1
    #     elif key == 'masks':
    #         i = 0
    #         for mask in rgbd_data[key]:
    #             cv2.imwrite(dirs + f"mask_{i + 1}-{j + 1}.jpg", mask)
    #             i += 1
    #     elif key == 'crops':
    #         for crop in rgbd_data[key]:
    #             print(f"rgbd: crop {crop}")
print(
    f"{time_sum[0] / test_num:.3f}, {time_sum[1] / test_num:.3f}, {time_sum[2] / test_num:.3f}, {time_sum[3] / test_num:.3f}")
print(
    f"{1 / time_sum[0] * test_num:.1f}, {1 / time_sum[1] * test_num:.1f}, {1 / time_sum[2] * test_num:.1f}, {1 / time_sum[3] * test_num:.1f}")
# recevier.close()
# sender.close()

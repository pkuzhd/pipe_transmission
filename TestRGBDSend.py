import cv2
import os
import numpy as np

from utils.RGBDSender import RGBDData, RGBDSender

if not os.path.exists("./test_dir"):
    os.mkdir("./test_dir")
if not os.path.exists("./pipe_dir"):
    os.mkdir("./pipe_dir")
if not os.path.exists("./pipe_dir/pipe2"):
    os.mkfifo("./pipe_dir/pipe2")

sender = RGBDSender()
sender.open("./pipe_dir/pipe2")

for i in range(5):
    imgs = [cv2.imread(f"./test_data/{(j + i) % 5 + 1}.jpg") for j in range(5)]
    h, w, _ = imgs[0].shape
    x, y = 0, 0
    depths = [np.zeros((h, w), dtype=np.float32) for j in range(5)]
    masks = [np.ones((h, w), dtype=np.uint8) for j in range(5)]
    data = RGBDData(5, imgs, depths, masks, [(w, h, 0, 0) for j in range(5)])
    sender.sendData(data)

sender.close()

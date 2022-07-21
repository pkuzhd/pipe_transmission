import cv2
import os
import numpy as np
import datetime

from utils.RGBDSender import RGBDData, RGBDSender

if not os.path.exists("./test_dir"):
    os.mkdir("./test_dir")
if not os.path.exists("./pipe_dir"):
    os.mkdir("./pipe_dir")
if not os.path.exists("./pipe_dir/pipe2"):
    os.mkfifo("./pipe_dir/pipe2")

sender = RGBDSender()
sender.open("./pipe_dir/pipe2")

i = 0
imgs = [cv2.imread(f"./test_data/{(j + i) % 5 + 1}.jpg") for j in range(5)]
h, w, _ = imgs[0].shape
x, y = 0, 0
depths = [np.zeros((h, w), dtype=np.float32) for j in range(5)]
masks = [np.ones((h, w), dtype=np.uint8) for j in range(5)]
data = RGBDData(5, imgs, depths, masks, [(w, h, 0, 0) for j in range(5)])

time_total = 0
bytes_total = 0
f = open("time_cost/1000_data_1.txt", "w")

for i in range(1000):
    start_time = datetime.datetime.now()
    bytes = sender.sendData(data)
    end_time = datetime.datetime.now()
    time_cost = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
    bytes_total += bytes
    time_total += time_cost

    print(f"{i} " + f"{time_cost} " + "ms " + "total: " + f"{time_total} " + "ms")
    f.write(f"{i} " + f"{time_cost} " + "ms " + "total: " + f"{time_total} " + "ms\n")
    print(f"{bytes_total / 1000 / time_total}" + " MBps\n")

print(bytes_total / 1000000, "MB")
f.write(f"{bytes_total / 1000000}" + " " + "MB\n")
print(time_total / 1000, "s")
f.write(f"{time_total / 1000}" + " s\n")
print(bytes_total / 1000 / time_total, "MBps")
f.write(f"{bytes_total / 1000 / time_total}" + " MBps\n")
print(f"{bytes_total / 1000 / time_total:.3f}",f"({time_total / 1000:.2f})")
f.close()
input()

sender.close()

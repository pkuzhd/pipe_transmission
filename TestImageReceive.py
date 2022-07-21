import cv2
import time
from utils.ImageReceiver import ImageData, ImageReceiver
import datetime
recv = ImageReceiver()

recv.open("/home/zhd/CLionProjects/pipe_transmission/pipe_dir/pipe1")
j = 0
while j < 1000:
    start_time = datetime.datetime.now()
    data = recv.getData()
    end_time = datetime.datetime.now()
    time_cost = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
    print(j," ",time_cost)
    time1 = time.time()
    for i in range(5):
        cv2.imwrite("/home/zhd/CLionProjects/pipe_transmission/saveData/img" + str(j)+"."+str(i+1)+".jpg",data.imgs[i])
    print(data)
    j += 1

input()


recv.close()

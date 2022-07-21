import cv2
import os
import numpy as np
import fcntl
import datetime
import time

class ImageData():
    def __init__(self, N, imgs):
        self.N = N
        self.imgs = imgs


class ImageReceiver():
    def __init__(self):
        self.rf = 0

    def open(self, filename):
        if not os.path.exists(filename):
            os.mkfifo(filename)
        self.rf = os.open(filename, os.O_RDONLY)
        print("os.open finished")
        print("pipe size", fcntl.fcntl(self.rf, 1032))
        fcntl.fcntl(self.rf, 1031, 1048576)
        print("pipe size", fcntl.fcntl(self.rf, 1032))
        
    def close(self):
        os.close(self.rf)

    def getData(self) -> ImageData:
        
        start_time = datetime.datetime.now()
        
        image_data = ImageData(0, [])
        end_time = datetime.datetime.now()
        time_cost = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
        
        print("init time", time_cost)
        
        start_time = datetime.datetime.now()
        time1 = time.time()
        buf = os.read(self.rf, 4)
        print("[1 bytes]" , time.time() - time1)
        end_time = datetime.datetime.now()
        time_cost = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
        print("read 1 byte time", time_cost)
        
        start_time = datetime.datetime.now()
        image_data.N = int.from_bytes(buf, byteorder='little', signed=False)
        # print(image_data.N)
        end_time = datetime.datetime.now()
        time_cost = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
        print("get num time", time_cost)
        
        start_time = datetime.datetime.now()
        

        for i in range(image_data.N):
            buf = os.read(self.rf, 4)
            w = int.from_bytes(buf, byteorder='little', signed=False)
            buf = os.read(self.rf, 4)
            h = int.from_bytes(buf, byteorder='little', signed=False)
            #print(w, h)

            buf = b''
            read_len = 0
            i = 0
            
            start_time = datetime.datetime.now()
            
            
            
            while read_len + 1048576 < w * h * 3:
                buf += os.read(self.rf, 1048576)
                read_len = np.frombuffer(buf, np.uint8).size
                
            end_time = datetime.datetime.now()
            time_cost = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
            
            # print("1048576:", time_cost)
            
            
            start_time = datetime.datetime.now()
            
            buf += os.read(self.rf, w * h * 3 - read_len)
            
            end_time = datetime.datetime.now()
            time_cost = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
            
            # print("remain:", time_cost)
            
            
            image_data.imgs.append(np.frombuffer(buf, np.uint8).reshape((h, w, 3)))
        
        end_time = datetime.datetime.now()
        time_cost = ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
        print(" for time ", time_cost)
        
        return image_data

    




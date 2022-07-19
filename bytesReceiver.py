import os
import time
import datetime

read_path = "pipe3"
if not os.path.exists(read_path):
    os.mkfifo(read_path)
 
rf = os.open(read_path, os.O_RDONLY)
print("os.open finished")

time_total = 0
byte_total = 0
while True:
    start_time = datetime.datetime.now()
    data = os.read(rf,10000000)
    print(len(data))
    end_time = datetime.datetime.now()
    time_total += ((end_time - start_time).seconds * 1000 + (end_time - start_time).microseconds / 1000)
    byte_total += data.size()
    print(byte_total, "B", time_total, "ms", byte_total / time_total / 1000, "MB/s")
    
    


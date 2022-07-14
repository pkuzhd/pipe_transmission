import cv2

from utils.ImageReceiver import ImageData, ImageReceiver

recv = ImageReceiver()

recv.open("./pipe_dir/pipe1")

for i in range(5):
    data = recv.getData()
    for j in range(5):
        cv2.imwrite(f"./test_dir/{j+1}-{i+1}.jpg", data.imgs[j])

recv.close()
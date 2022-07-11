import cv2
from utils.ImageReceiver import ImageData, ImageReceiver
from utils.RGBDSender import RGBDData, RGBDSender


def func(data: ImageData) -> RGBDData:
    pass


imgs = [cv2.imread(f"./test_data/{j + 1}.fg.jpg") for j in range(5)]

for i in range(5):
    data = ImageData(5, imgs)

    rgbd_data = func(data)

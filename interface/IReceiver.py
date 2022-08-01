import numpy as np

from typing import List


class Frames:
    width: List[int]
    height: List[int]
    channel: List[int]
    id: List[int]
    pts: np.float64
    data: np.ndarray  # dtype = np.uint8, shape = (n, h, w, c)


class IFrameReceiver:
    def recvFrames(self) -> Frames:
        raise NotImplementedError


class RGBDMData:
    images: np.ndarray  # dtype = np.uint8, shape = (n, h, w, c)
    depths: np.ndarray  # dtype = np.float32, shape = (n, h_, w)
    masks: np.ndarray  # dtype = np.uint8, shape = (n, h, w)
    crops: np.ndarray  # (w, h, x, y), dtype = np.uint32, shape = (n, 4)
    pts: np.float64


class IDepthEstimator:
    def estimate(self, frames: Frames) -> RGBDMData:
        raise NotImplementedError


class IRGBDMSender:
    def sendRGBDM(self, data: RGBDMData) -> None:
        raise NotImplementedError

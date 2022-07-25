from MultiProcessBuffer import MultiProcessBuffer

import depth_estimation.DepthEstimation

class DepthProcess():

    def __init__(self,
                 in_buffer: MultiProcessBuffer,
                 out_buffer: MultiProcessBuffer):
        self.in_buffer = in_buffer
        self.out_buffer = out_buffer

    def run(self):
        pass

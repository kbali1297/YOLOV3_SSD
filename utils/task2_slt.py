"""
Goal of Task 2:
    Implement a helper function to transform the label format into the required format in YOLOv3.
"""


import numpy as np


def xywh2xyxy_np_slt(xywh):
    """
    input:
        xywh (type: np.ndarray, shape: (n,4), dtype: int16): n bounding boxes with the xywh format (center based)

    output:
        xyxy (type: np.ndarray, shape: (n,4), dtype: int16): n bounding boxes with the xyxy format (edge based)
    """

    # Solution:
    ########################
    #  Start of your code  #
    ########################

    xyxy = np.zeros_like(xywh)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2

    ########################
    #   End of your code   #
    ########################

    return xyxy


if __name__ == "__main__":
    # Execute this file to check your output of this example
    xywh_example = np.asarray([[150, 120, 20, 10], [258, 89, 55, 45]], dtype=np.int16)
    your_xyxy = xywh2xyxy_np_slt(xywh_example)
    print(f"Your xyxy: {your_xyxy}")
    print(f"Your xyxy shape: {your_xyxy.shape}")
    print(f"Your xyxy dtype: {your_xyxy.dtype}")

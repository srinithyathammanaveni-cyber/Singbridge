"""
preprocessing.py
────────────────
Image preprocessing pipeline for hand gesture recognition.
"""

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, target_size=(640, 480)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.frame_count = 0

    def process(self, frame):
        if frame is None:
            return None

        frame = cv2.resize(frame, self.target_size)
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        self.frame_count += 1
        return sharpened

    def get_skin_mask(self, frame):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_y = np.array([0,   133, 77],  dtype=np.uint8)
        upper_y = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_y, upper_y)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0,  20,  70],  dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        combined = cv2.bitwise_or(mask_ycrcb, mask_hsv)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)
        combined = cv2.GaussianBlur(combined, (3, 3), 0)

        return combined

# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:24:53 2024

@author: jacob
"""

import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

im = cv2.imread('zodiac.jpg')
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
plt.show()

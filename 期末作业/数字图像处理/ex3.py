import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage import exposure
import math
import os ,sys
from rel import RLE



def show(img, name='img'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_plt(img):
    plt.imshow(img, 'gray')
    plt.show()

def calcu_code(path):

imgs = ['./gray1.jpg', './gray2.jpg']

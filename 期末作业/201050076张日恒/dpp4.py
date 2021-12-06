
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import math
from selffunc import *
from jpg import JPEGEncode
from rel import  RLE
from huffman import  Huffman

j = JPEGEncode()
j.compress(0.2)
j.compress(0.6)
j.compress(0.8)
rle = RLE()
rle.compress()
h = Huffman()
h.compress(8)
h.compress(16)
h.compress(32)


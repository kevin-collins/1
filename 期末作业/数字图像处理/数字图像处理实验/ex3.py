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


def compress(path, q_factor):
    image = cv2.imread(path, 1)
    # Step 1: convert rgb image space tp YCrCb space
    image =  cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # 图像尺寸调整，以适应分块
    height, width = image.shape[:2]
    if height % 8 != 0 or width % 8 != 0:
        image = np.pad(image, ((0, (8 - height % 8) % 8), (0, (8 - width % 8) % 8), (0, 0)),
                       "edge")
    height, width = image.shape[:2]
    size = sys.getsizeof((image.flatten()))

    print("Image {}:".format(path))
    print("Origin Image's Size is {:.2f} KB.".format(size / 1024))

    [y, cr, cb] = cv2.split(image)
    # Step 2: DCT decomposition, transform from time-domain to
    # frequency-domain, and choose 8*8 block
    image_dct = []
    for img in [y, cr, cb]:
        f_patches = []
        fi_patches = []
        # 图像分块
        h_patches = np.vsplit(img, height // 8)
        for i in range(height // 8):
            wh_patches = np.hsplit(h_patches[i], width // 8)
            f_patch = []
            fi_patch = []
            for j in range(width // 8):
                # DCT 变换
                patch_dct = cv2.dct(wh_patches[j].astype(np.float))
                f_patch.append(patch_dct)
            f_patchs = np.hstack(f_patch)
            f_patches.append(f_patchs)
        img_dct = np.vstack(f_patches)
        image_dct.append(img_dct)

    image_dct = np.moveaxis(image_dct, 0, 2)

    # Step 3: 量化
    image_dct = np.around(image_dct / q_factor)
    # Step 4: 行程编码，转换为一维数组
    rle = RLE()
    [d_y, d_cr, d_cb] = cv2.split(image_dct)
    image_rle = []
    for dct in [d_y, d_cr, d_cb]:
        dct_rle = rle.compressimg(dct)
        image_rle.append(dct_rle)

    # 图像大小计算，压缩比计算
    r_size = sys.getsizeof((image_rle))
    print("quality factor:{:.2f}".format(q_factor))
    print("After Run JPEG Compress Image's Size is  {:.2f} KB.\
            \nCompressed Image's size is {:.4%} of Origin Image.".
          format(r_size / 1024, r_size / size))

    image_iq = image_dct * q_factor
    [r_y, r_cr, r_cb] = cv2.split(image_iq)
    image_back = []
    for img in [r_y, r_cr, r_cb]:
        f_patches = []
        # 图像分块
        h_patches = np.vsplit(img, height // 8)
        for i in range(height // 8):
            wh_patches = np.hsplit(h_patches[i], width // 8)
            f_patch = []
            fi_patch = []
            for j in range(width // 8):
                # IDCT 变换
                patch_dct = cv2.idct(wh_patches[j].astype(np.float))
                f_patch.append(patch_dct)
            f_patchs = np.hstack(f_patch)
            f_patches.append(f_patchs)
        img_back = np.vstack(f_patches).astype(np.uint8)
        image_back.append(img_back)
    image_back = np.moveaxis(image_back, 0, 2)

    # YCrCb 空间转换回 RGB 空间
    image_back = cv2.cvtColor(image_back, cv2.COLOR_YCrCb2RGB)

    show_plt(image_back)
    mse = ((image - image_back)**2).mean()
    print("Compressed Image's MSE is {:.2f}".format(mse))



imgs = ['./img1.jpg', './img2.jpg']
for img in imgs:
    compress(img, 0.2)
    compress(img, 0.6)
    compress(img, 0.8)

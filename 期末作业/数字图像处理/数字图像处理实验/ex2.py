import cv2
import numpy as np
import matplotlib.pyplot as plt
def to_gray(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def show(img, name='img'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_plt(img):
    plt.imshow(img, 'gray')
    plt.show()

# 傅里叶变换
def DFT(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    fshift = np.fft.fftshift(dft)

    return fshift


# 高通滤波器
def high(path):
    img = to_gray(path)
    rows, cols = img.shape

    fshift = DFT(img)


    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

    mask = np.ones((rows, cols, 2), np.uint8)

    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # 掩膜图像和频谱图像乘积

    f = fshift * mask
    ori = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    result = 20 * np.log(cv2.magnitude(f[:,:, 0], f[:, :, 1]))

    # plt.imshow(ori, cmap="gray")
    plt.imshow(result, cmap="gray")

    # plt.savefig("result.jpg", dpi = 300, bbox_inches = "tight")
    plt.show()
    # 傅里叶逆变换

    ishift = np.fft.ifftshift(f)

    iimg = cv2.idft(ishift)

    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    # print(res)
    show_plt(res)


# 低通滤波器
def low(path):
    img = to_gray(path)
    rows, cols = img.shape

    fshift = DFT(img)

    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

    mask = np.zeros((rows, cols, 2), np.uint8)

    mask[crow - 100:crow + 100, ccol - 100:ccol + 100] = 1

    # 掩膜图像和频谱图像乘积

    f = fshift * mask

    ori = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    result = 20 * np.log(cv2.magnitude(f[:,:, 0], f[:, :, 1]))

    # plt.imshow(ori, cmap="gray")
    plt.imshow(result, cmap="gray")

    # plt.savefig("result.jpg", dpi = 300, bbox_inches = "tight")
    plt.show()
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)

    iimg = cv2.idft(ishift)

    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    show_plt(res)


gray_imgs = ['./gray1.jpg', './gray2.jpg']

"""
for img in gray_imgs:
    high(img)
"""
def dct(img):
    C_temp = np.zeros((8, 8))
    C_temp[0, :] = 1 * np.sqrt(1 / 8)

    for i in range(1, 8):
        for j in range(8):
            C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)
                                  ) * np.sqrt(2 / 8)

    dst = np.dot(C_temp, img)

    dst = np.dot(dst, np.transpose(C_temp))
    t = dst[0][0]
    for i in range(len(dst)):
        for j in range(len(dst[0])):
            dst[i][j] = 0
    dst[0][0] = t
    dst1 = np.log(abs(dst))  # 进行log处理

    img_recor = np.dot(np.transpose(C_temp), dst)
    img_recor1 = np.dot(img_recor, C_temp)
    return dst1, img_recor1

def dct_gray(path):

    image = to_gray(path)
    height, width = image.shape
    if height % 8 != 0 or width % 8 != 0:
        image = np.pad(image, ((0, (8 - height % 8) % 8),(0, (8 - width % 8) % 8)),"edge")
    height, width = image.shape
    f_patches = []
    fi_patches = []
    h_patches = np.vsplit(image, height // 8)

    for i in range(height // 8):
        wh_patches = np.hsplit(h_patches[i], width // 8)
        f_patch = []
        fi_patch = []
        for j in range(width // 8):
            # DCT 变换
            patch_dct, patch_idct = dct(wh_patches[j].astype(np.float))
            # IDCT 变换
            # patch_idct = cv2.idct(patch_dct)
            f_patch.append(patch_dct)
            fi_patch.append(patch_idct)
        f_patchs = np.hstack(f_patch)
        f_patches.append(f_patchs)
        fi_patchs = np.hstack(fi_patch)
        fi_patches.append(fi_patchs)
    image_dct = np.vstack(f_patches)
    image_back = np.vstack(fi_patches).astype(np.uint8)

    show(image_dct, 'dct')
    show(image_back, 'restore')


for img in gray_imgs:
    dct_gray(img)



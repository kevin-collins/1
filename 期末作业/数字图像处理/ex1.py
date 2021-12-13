import cv2
import numpy as np
import random
"""
实验1
"""
gray_imgs = ['./gray1.jpg', './gray2.jpg']
rgb_imgs = ['./img1.jpg', './img2.jpg']

def to_gray(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color(img_gray):
    row, col = img_gray.shape[:]
    b = np.zeros((row, col))
    g = np.zeros((row, col))
    r = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if img_gray[i, j] < 255//4:
                g[i, j] = 255
                r[i, j] = img_gray[i, j] * 4
                while r[i, j] > 255:
                    r[i, j] -= 255
                b[i, j] = 0
            elif img_gray[i, j] < 255//2:
                g[i, j] = 255
                b[i, j] = img_gray[i, j] * 2
                while b[i, j] > 255:
                    b[i, j] -= 255
                r[i, j] = 0
            elif(img_gray[i, j]<3*255//4):
                b[i, j] = 255
                r[i, j] = img_gray[i, j] * 3 // 4
                while r[i, j] > 0:
                    r[i, j] -= 255
                g[i, j] = 0
            else:
                b[i, j] = 255
                g[i, j] = img_gray[i, j]
                while g[i, j] > 255:
                    g[i, j] -= 255
                r[i, j] = 0
    img_color = cv2.merge([b, g, r])
    return img_color


def gray2rgb(path):
    img_gray = to_gray(path)
    img_color = color(img_gray)
    show(img_gray)
    show(img_color)


def guassian_noise(image, mean=0, var=0.02):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var**0.5)
    out = image + noise
    if out.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def salt_pepper_noise(image, prob):   # prob:噪声比例
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def add_noise(path):
    image = to_gray(path)
    # img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # show(image)
    img_gs = guassian_noise(image)
    img_jy = salt_pepper_noise(image, 0.2)

    # show(img_gs)
    # show(img_jy)
    return img_gs, img_jy


def gaussian_filter(img, K_size=3, sigma=1.3):

    if len(img.shape) == 3:

        H, W, C = img.shape

    else:

        img = np.expand_dims(img, axis=-1)

        H, W, C = img.shape

    ## Zero padding

    pad = K_size // 2

    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)

    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    ## prepare Kernel

    K = np.zeros((K_size, K_size), dtype=np.float)

    for x in range(-pad, -pad + K_size):

        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    K /= (2 * np.pi * sigma * sigma)

    K /= K.sum()

    tmp = out.copy()

    # filtering

    for y in range(H):

        for x in range(W):

            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

    out = np.clip(out, 0, 255)

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    show(out)
    return out


for img in gray_imgs:
    img_gs, img_jy = add_noise(img)
    out = cv2.GaussianBlur(img_gs, (7, 7), 0, 0)
    show(out)
    out = cv2.GaussianBlur(img_jy, (7, 7), 0, 0)
    show(out)

#Roberts算子
def sobel(path):
    img = to_gray(path)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0) #对x求一阶导
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1) #对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    img_s = img + cv2.addWeighted(absX, 0.5, absY, -0.5, 0) + cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    show(img_s)
    return img_s
for img in gray_imgs:
    sobel(img)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage import exposure
import math
def equalizehist() :
    outdir = './result1/'
    for i in range(5):
        i = i + 1
        imgpath = 'img' + str(i) + '.jpg'
        image = cv2.imread(imgpath,1)
        r=image[:,:,2]
        g=image[:,:,1]
        b=image[:,:,0]
        r1 = r.flatten()
        g1 = g.flatten()
        b1 = b.flatten()

        outpath = outdir + 'hist' + str(i) + '.jpg'
        plt.title(outpath)
        plt.hist(r1,bins=256,density=1,facecolor='red')
        plt.hist(g1,bins=256,density=1,facecolor='green')
        plt.hist(b1,bins=256,density=1,facecolor='blue')
        plt.savefig(outpath)
        plt.close()


        r2 = cv2.equalizeHist(r)
        g2 = cv2.equalizeHist(g)
        b2 = cv2.equalizeHist(b)
        img_new = cv2.merge((b2,g2,r2))
        outimg = outdir + 'ehimg' + str(i) + '.jpg'
        cv2.imwrite(outimg,img_new)

        outpath = outdir + 'equalizehist' + str(i) + '.png'
        plt.title(outpath)
        r2 = r2.flatten()
        g2 = g2.flatten()
        b2 = b2.flatten()
        plt.hist(r2,bins=256,density=1,facecolor='red')
        plt.hist(g2,bins=256,density=1,facecolor='green')
        plt.hist(b2,bins=256,density=1,facecolor='blue')
        plt.savefig(outpath)
        plt.close()

def rgbtohsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv2.split(rgb_lwpImg)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv2.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            # print(num)
            # print(den)
            theta = float(np.arccos(num/den))

            if den == 0:
                    H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2*3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3*min_RGB/sum

            H = H/(2*3.14159265)
            I = sum/3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H*255
            hsi_lwpImg[i, j, 1] = S*255
            hsi_lwpImg[i, j, 2] = I*255
    return hsi_lwpImg

def hsitorgb(hsi_img):
    h = int(hsi_img.shape[0])
    w = int(hsi_img.shape[1])
    H, S, I = cv2.split(hsi_img)
    H = H / 255.0
    S = S / 255.0
    I = I / 255.0
    bgr_img = hsi_img.copy()
    B, G, R = cv2.split(bgr_img)
    for i in range(h):
        for j in range(w):
            if S[i, j] < 1e-6:
                R = I[i, j]
                G = I[i, j]
                B = I[i, j]
            else:
                H[i, j] *= 360
                if H[i, j] > 0 and H[i, j] <= 120:
                    B = I[i, j] * (1 - S[i, j])
                    R = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    G = 3 * I[i, j] - (R + B)
                elif H[i, j] > 120 and H[i, j] <= 240:
                    H[i, j] = H[i, j] - 120
                    R = I[i, j] * (1 - S[i, j])
                    G = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    B = 3 * I[i, j] - (R + G)
                elif H[i, j] > 240 and H[i, j] <= 360:
                    H[i, j] = H[i, j] - 240
                    G = I[i, j] * (1 - S[i, j])
                    B = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    R = 3 * I[i, j] - (G + B)
            bgr_img[i, j, 0] = B * 255
            bgr_img[i, j, 1] = G * 255
            bgr_img[i, j, 2] = R * 255
    return bgr_img

def rgb2hsi2rgb():
    outdir = './result1/'
    for i in range(5):
        i = i + 1
        imgpath = 'img' + str(i) + '.jpg'
        image = cv2.imread(imgpath)
        hsi = rgbtohsi(image)
        outimg = outdir + 'hsi' + str(i) + '.jpg'
        cv2.imwrite(outimg, hsi)
        h = hsi[:, :, 0]
        s = hsi[:, :, 1]
        inten = hsi[:, :, 2]
        i2 = cv2.equalizeHist(inten)

        img_hsi = cv2.merge((h, s, i2))
        img_rgb = hsitorgb(img_hsi)
        outimg = outdir + 'eh_img_i' + str(i) + '.jpg'
        cv2.imwrite(outimg, img_rgb)

        i2 = i2.flatten()
        outpath = outdir + 'equ_h_i' + str(i) + '.png'
        plt.hist(i2, bins=256, density=1, facecolor='blue')
        plt.savefig(outpath)
        plt.close()

def guassian_noise(image, mean=0, var=0.01):
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

def add_noise():
    for i in range(5):
        i = i + 1
        imgpath = 'img' + str(i) + '.jpg'
        image = cv2.imread(imgpath)
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gs = guassian_noise(img_color)
        outimg = 'img_gs' + str(i) + '.jpg'
        cv2.imwrite(outimg, img_gs)
        img_jy = salt_pepper_noise(img_color, 0.2)
        outimg = 'img_jy' + str(i) + '.jpg'
        cv2.imwrite(outimg, img_jy)

def de_noise():
    for i in range(5):
        i = i + 1
        imgpath = 'img' + str(i) + '.jpg'
        image = cv2.imread(imgpath)
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gs = guassian_noise(img_color)
        img_jy = salt_pepper_noise(img_color, 0.2)

        #均值滤波去噪
        outimg1 = 'gs_blur_5_' + str(i) + '.jpg'
        outimg2 = 'jy_blur_5_' + str(i) + '.jpg'
        result1 = cv2.blur(img_gs, (5, 5))
        result2 = cv2.blur(img_jy, (5, 5))
        cv2.imwrite(outimg1, result1)
        cv2.imwrite(outimg2, result2)

        outimg1 = 'gs_blur_3_' + str(i) + '.jpg'
        outimg2 = 'jy_blur_3_' + str(i) + '.jpg'
        result1 = cv2.blur(img_gs, (3, 3))
        result2 = cv2.blur(img_jy, (3, 3))
        cv2.imwrite(outimg1, result1)
        cv2.imwrite(outimg2, result2)

        outimg1 = 'gs_blur_7_' + str(i) + '.jpg'
        outimg2 = 'jy_blur_7_' + str(i) + '.jpg'
        result1 = cv2.blur(img_gs, (7, 7))
        result2 = cv2.blur(img_jy, (7, 7))
        cv2.imwrite(outimg1, result1)
        cv2.imwrite(outimg2, result2)

        #中值滤波去噪
        outimg1 = 'gs_meblur_3_' + str(i) + '.jpg'
        outimg2 = 'jy_meblur_3_' + str(i) + '.jpg'
        result1 = cv2.medianBlur(img_gs, 3)
        result2 = cv2.medianBlur(img_jy, 3)
        cv2.imwrite(outimg1, result1)
        cv2.imwrite(outimg2, result2)

        outimg1 = 'gs_meblur_5_' + str(i) + '.jpg'
        outimg2 = 'jy_meblur_5_' + str(i) + '.jpg'
        result1 = cv2.medianBlur(img_gs, 5)
        result2 = cv2.medianBlur(img_jy, 5)
        cv2.imwrite(outimg1, result1)
        cv2.imwrite(outimg2, result2)

        outimg1 = 'gs_meblur_7_' + str(i) + '.jpg'
        outimg2 = 'jy_meblur_7_' + str(i) + '.jpg'
        result1 = cv2.medianBlur(img_gs, 7)
        result2 = cv2.medianBlur(img_jy, 7)
        cv2.imwrite(outimg1, result1)
        cv2.imwrite(outimg2, result2)
        #高斯滤波去噪

        outimg1 = 'gs_gublur_3_' + str(i) + '.jpg'
        outimg2 = 'jy_gublur_3_' + str(i) + '.jpg'
        result1 = cv2.GaussianBlur(img_gs, (3, 3), 0, 0)
        result2 = cv2.GaussianBlur(img_jy, (3, 3), 0, 0)
        cv2.imwrite(outimg1, result1)
        cv2.imwrite(outimg2, result2)

        outimg1 = 'gs_gublur_5_' + str(i) + '.jpg'
        outimg2 = 'jy_gublur_5_' + str(i) + '.jpg'
        result1 = cv2.GaussianBlur(img_gs, (5, 5), 0, 0)
        result2 = cv2.GaussianBlur(img_jy, (5, 5), 0, 0)
        cv2.imwrite(outimg1, result1)
        cv2.imwrite(outimg2, result2)

        outimg1 = 'gs_gublur_7_' + str(i) + '.jpg'
        outimg2 = 'jy_gublur_7_' + str(i) + '.jpg'
        result1 = cv2.GaussianBlur(img_gs, (7, 7), 0, 0)
        result2 = cv2.GaussianBlur(img_jy, (7, 7), 0, 0)
        cv2.imwrite(outimg1, result1)
        cv2.imwrite(outimg2, result2)

    return  0

def SobelAlogrithm():

    outdir = './result1/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        img = cv2.imread(imgpath, 0)

        r = img.shape[0]
        c = img.shape[1]
        new_image = np.zeros((r, c))
        new_imageX = np.zeros(img.shape)
        new_imageY = np.zeros(img.shape)
        s_suanziX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
        s_suanziY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        for i in range(r - 2):
            for j in range(c - 2):
                new_imageX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziX))
                new_imageY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * s_suanziY))
                new_image[i + 1, j + 1] = (new_imageX[i + 1, j + 1] * new_imageX[i + 1, j + 1] + new_imageY[i + 1, j + 1] *
                                           new_imageY[i + 1, j + 1]) ** 0.5

        new_image = np.uint8(new_image)
        outimg1 = outdir + 'Sobel' + str(index) + '.jpg'
        outimg2 = outdir + 'origin+Sobel' + str(index) + '.jpg'

        orig_sobel = img + new_image
        cv2.imwrite(outimg1, new_image)
        cv2.imwrite(outimg2, orig_sobel)


def LaplaceAlogrithm():
    outdir = './result1/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        img = cv2.imread(imgpath, 0)
        r = img.shape[0]
        c = img.shape[1]
        new_image = np.zeros((r, c))
        L_suanzi = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        # L_suanzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        for i in range(r - 2):
            for j in range(c - 2):
                new_image[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * L_suanzi))

        new_image = np.uint8(new_image)
        outimg1 = outdir + 'Laplace' + str(index) + '.jpg'
        outimg2 = outdir + 'origin+Laplace' + str(index) + '.jpg'

        orig_laplace = img + new_image
        cv2.imwrite(outimg1, new_image)
        cv2.imwrite(outimg2, orig_laplace)

def mineAlogrithm():
    outdir = './result1/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        img = cv2.imread(imgpath, 0)
        r = img.shape[0]
        c = img.shape[1]
        new_image = np.zeros((r, c))
        mine_suanzi = np.array([[-1, 0, -1], [0, 4, 0], [-1, 0, -1]])
        # L_suanzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        for i in range(r - 2):
            for j in range(c - 2):
                new_image[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * mine_suanzi))

        new_image = np.uint8(new_image)
        outimg1 = outdir + 'mine' + str(index) + '.jpg'
        outimg2 = outdir + 'origin+mine' + str(index) + '.jpg'

        orig_mine = img + new_image
        cv2.imwrite(outimg1, new_image)
        cv2.imwrite(outimg2, orig_mine)

def lowpass_fliter():
    outdir = './result2/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        image = cv2.imread(imgpath)
        # img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_color =  image
        height, width, channel = img_color.shape
        # 低通
        mask_low = np.zeros((img_color.shape[0], img_color.shape[1]), np.uint8)
        img = np.zeros(img_color.shape)
        f = np.zeros(img_color.shape)
        raduis_list = [5,20,50,80,250]
        for raduis in raduis_list:
            for i in range(3):
                mask_low[int(height / 2) - raduis:int(height / 2) + raduis, int(width / 2) - raduis:int(width / 2) + raduis] = 1
                f1 = np.fft.fft2(img_color[:, :, i])
                f1shift = np.fft.fftshift(f1)
                f1shift = f1shift * mask_low
                f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
                img_new = np.fft.ifft2(f2shift)
                f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
                img_new = np.fft.ifft2(f2shift)
                # 出来的是复数，无法显示
                img_new = np.abs(img_new)

                # img_idf = cv2.idft(img_idf)
                # img_new = 20 * np.log(np.abs(img_new))
                # 20 * np.log(np.abs(fshift))
                # img_new = cv2.idft(img_new)
                # img_new = cv2.magnitude(img_new[:, :, 0], img_new[:, :, 1])
                # img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
                img[:, :, i] = img_new
                f[:, :, i] = f1shift

            outimg1 = outdir + 'lowpass_spatial_'  + str(raduis) + '_' + str(index) + '.jpg'
            outimg2 = outdir + 'lowpass_frequency_' + str(raduis) + '_' + str(index) + '.jpg'
            cv2.imwrite(outimg1, img)
            cv2.imwrite(outimg2, f)

def highpass_fliter():
    outdir = './result2/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        image = cv2.imread(imgpath)
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channel = img_color.shape
        # 高通
        mask_high = np.ones((img_color.shape[0], img_color.shape[1]), np.uint8)
        img = np.zeros(img_color.shape)
        f = np.zeros(img_color.shape)
        raduis_list = [5, 20, 50]
        for raduis in raduis_list:
            for i in range(3):
                mask_high[int(height / 2) - raduis:int(height / 2) + raduis, int(width / 2) - raduis:int(width / 2) + raduis] = 0
                f1 = np.fft.fft2(img_color[:, :, i])
                f1shift = np.fft.fftshift(f1)
                f1shift = f1shift * mask_high
                f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
                img_new = np.fft.ifft2(f2shift)
                img_new = np.abs(img_new)

                # img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
                img[:, :, i] = img_new
                f[:, :, i] = f1shift

            outimg1 = outdir + 'highpass_spatial_' + str(raduis) + '_' + str(index) + '.jpg'
            outimg2 = outdir + 'highpass_frequency_' + str(raduis) + '_' + str(index) + '.jpg'
            cv2.imwrite(outimg1, img)
            cv2.imwrite(outimg2, f)

def fft_gray():
    outdir = './result2/'
    for i in range(5):
        i = i + 1
        imgpath = 'img' + str(i) + '.jpg'
        image = cv2.imread(imgpath,0)
        f = np.fft.fft2(image)  # 快速傅里叶变换算法得到频率分布
        fshift = np.fft.fftshift(f)  # 默认结果中心点位置是在左上角，转移到中间位置
        img_dft = 20 * np.log(np.abs(fshift))  # 结果是复数，求绝对值才是振幅

        f1shift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f1shift)
        img_back = np.abs(img_back)

        outimg1 = outdir + 'gray' + str(i) + '.jpg'
        outimg2 = outdir + 'fft_result' + str(i) + '.jpg'
        outimg3 = outdir + 'gray_back' + str(i) + '.jpg'

        cv2.imwrite(outimg1, image)
        cv2.imwrite(outimg2, img_dft)
        cv2.imwrite(outimg3, img_back)

def fft_color():
    outdir = './result2/'
    for i in range(5):
        i = i + 1
        imgpath = 'img' + str(i) + '.jpg'
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_f = np.zeros(image.shape)
        image_b = np.zeros(image.shape)
        for c in range(3):
            f = np.fft.fft2(image[:,:,c])
            fshift = np.fft.fftshift(f)
            img_dft = 20 * np.log(np.abs(fshift))
            image_f[:,:,c]=img_dft

            f1shift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f1shift)
            img_back = np.abs(img_back)

            image_b[:, :, c] = img_dft


        outimg1 = outdir + 'color' + str(i) + '.jpg'
        outimg2 = outdir + 'color_fft_result' + str(i) + '.jpg'
        outimg3 = outdir + 'color_back' + str(i) + '.jpg'

        cv2.imwrite(outimg1, image)
        cv2.imwrite(outimg2, image_f)
        cv2.imwrite(outimg3, img_back)

def dct_gray():
    outdir = './result2/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        image = cv2.imread(imgpath, 0)
        # print(image.shape)
        height, width = image.shape
        if height % 8 != 0 or width % 8 != 0:
            image = np.pad(image, ((0, (8 - height % 8) % 8),
                                   (0, (8 - width % 8) % 8)),
                           "edge")
        print(image.shape)
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
                patch_dct = cv2.dct(wh_patches[j].astype(np.float))
                # IDCT 变换
                patch_idct = cv2.idct(patch_dct)
                f_patch.append(patch_dct)
                fi_patch.append(patch_idct)
            f_patchs = np.hstack(f_patch)
            f_patches.append(f_patchs)
            fi_patchs = np.hstack(fi_patch)
            fi_patches.append(fi_patchs)
        image_dct = np.vstack(f_patches)
        image_back = np.vstack(fi_patches).astype(np.uint8)
        outimg1 = outdir + 'gray_dct' + str(index) + '.jpg'
        outimg2 = outdir + 'gray_dct_restore' + str(index) + '.jpg'

        cv2.imwrite(outimg1, image_dct)
        cv2.imwrite(outimg2, image_back)
        # plt.subplot(132), plt.imshow(image_dct), plt.title('DCT_result'), plt.axis('off')
        # plt.subplot(133), plt.imshow(image_back), plt.title('restored_img'), plt.axis('off')
        # plt.show()

def dct_color():
    outdir = './result2/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if height % 8 != 0 or width % 8 != 0:
            image = np.pad(image, ((0, (8 - height % 8) % 8),
                                   (0, (8 - width % 8) % 8), (0, 0))
                           , "edge")
        height, width = image.shape[:2]
        [b, g, r] = cv2.split(image)

        image_dct = []
        image_back = []

        for img in [b, g, r]:
            f_patches = []
            fi_patches = []
            h_patches = np.vsplit(img, height // 8)
            for i in range(height // 8):
                wh_patches = np.hsplit(h_patches[i], width // 8)
                f_patch = []
                fi_patch = []
                for j in range(width // 8):
                    # DCT 变换
                    patch_dct = cv2.dct(wh_patches[j].astype(np.float))
                    # IDCT 变换
                    patch_idct = cv2.idct(patch_dct)
                    f_patch.append(patch_dct)
                    fi_patch.append(patch_idct)
                f_patchs = np.hstack(f_patch)
                f_patches.append(f_patchs)
                fi_patchs = np.hstack(fi_patch)
                fi_patches.append(fi_patchs)
            img_dct = np.vstack(f_patches)
            img_back = np.vstack(fi_patches).astype(np.uint8)
            image_dct.append(img_dct)
            image_back.append(img_back)
        image_dct = np.moveaxis(image_dct, 0, 2)
        image_back = np.moveaxis(image_back, 0, 2)
        outimg1 = outdir + 'color_dct' + str(index) + '.jpg'
        outimg2 = outdir + 'color_dct_restore' + str(index) + '.jpg'

        cv2.imwrite(outimg1, image_dct)
        cv2.imwrite(outimg2, image_back)

    return 0

def sobel():
    outdir = './result3/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        new_image = np.zeros((height, width, 3))
        new_imageX = np.zeros(image.shape)
        new_imageY = np.zeros(image.shape)
        s_X = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # X方向
        s_Y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        for k in range(3):
            for i in range(height - 2):
                for j in range(width - 2):
                    new_imageX[i + 1, j + 1, k] = abs(np.sum(image[i:i + 3, j:j + 3, k] * s_X))
                    new_imageY[i + 1, j + 1, k] = abs(np.sum(image[i:i + 3, j:j + 3, k] * s_Y))

            new_image[:, :, k] = (new_imageX[:, :, k] * new_imageX[:, :, k]
                                  + new_imageY[:, :, k] * new_imageY[:, :,k]) ** 0.5

        new_image = np.uint8(new_image)
        outimg1 = outdir + 'sobel_' + str(index) + '.jpg'
        cv2.imwrite(outimg1, new_image)

def hough_line():
    outdir = './result3/'
    for index in range(5):
        index = index + 1
        imgpath = outdir + 'sobel_' + str(index) + '.jpg'
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 2)

        outimg1 = outdir + 'hough_' + str(index) + '.jpg'
        cv2.imwrite(outimg1, gray)

def watershed():
    outdir = './result3/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        image = cv2.imread(imgpath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3    DIST_L2 可以为3或者5
        ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknow = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)  # 对连通区域进行标号  序号为 0 - N-1
        markers = markers + 1
        markers[unknow == 255] = 0
        markers = cv2.watershed(image, markers)  # 分水岭算法后，所有轮廓的像素点被标注为  -1
        # print(markers)

        image[markers == -1] = [0, 0, 255]  # 标注为-1 的像素点标 红
        outimg1 = outdir + 'watershed_' + str(index) + '.jpg'
        cv2.imwrite(outimg1, image)

        # cv2.imshow("thresh", thresh)

def calcGrayHist(image):
    rows,cols = image.shape
    grayHist = np.zeros([256],np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] +=1#把图像灰度值作为索引
    return(grayHist)

def histpart():
    outdir = './result3/'
    for index in range(5):
        index = index + 1
        imgpath = 'img' + str(index) + '.jpg'
        image = cv2.imread(imgpath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(gray.shape)
        g = gray.flatten()
        outpath = outdir + 'grayhist' + str(index) + '.jpg'
        plt.title(outpath)
        plt.hist(g, bins=256, density=1, facecolor='blue')
        plt.savefig(outpath)
        plt.close()








# img_man = cv2.imread(r'C:\Users\ydh19\my_homework\img_1.jpg')  # 直接读为灰度图像
# img_man = cv2.cvtColor(img_man, cv2.COLOR_BGR2RGB)
# plt.subplot(131), plt.imshow(img_man), plt.title('original')
# plt.xticks([]), plt.yticks([])
#
# rows, cols, zzz = img_man.shape
# mask0 = np.zeros((img_man.shape[0], img_man.shape[1]), np.uint8)  # 低通
# mask1 = np.ones((img_man.shape[0], img_man.shape[1]), np.uint8)  # 高通
# img = np.zeros(img_man.shape)
# f = np.zeros(img_man.shape)
# print(f.shape)
#
# # 低通
# for i in range(3):
#     mask0[int(rows / 2) - 250:int(rows / 2) + 250, int(cols / 2) - 250:int(cols / 2) + 250] = 1
#     # --------------------------------
#     f1 = np.fft.fft2(img_man[:, :, i])
#     f1shift = np.fft.fftshift(f1)
#     f1shift = f1shift * mask0
#     f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
#     img_new = np.fft.ifft2(f2shift)
#     # 出来的是复数，无法显示
#     img_new = np.abs(img_new)
#     print(img_new.shape)
#     # 调整大小范围便于显示
#     img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
#     img[:, :, i] = img_new
#     f[:, :, i] = f1shift
# # 高通
# for i in range(3):
#     mask1[int(rows / 2) - 250:int(rows / 2) + 250, int(cols / 2) - 250:int(cols / 2) + 250] = 0
#     # --------------------------------
#     f1 = np.fft.fft2(img_man[:,:,i])
#     f1shift = np.fft.fftshift(f1)
#     f1shift = f1shift * mask1
#     f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
#     img_new = np.fft.ifft2(f2shift)
#     # 出来的是复数，无法显示
#     img_new = np.abs(img_new)
#     print(img_new.shape)
#     # 调整大小范围便于显示
#     img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
#     img[:,:,i] = img_new
#     f[:,:,i] = f1shift
#
# plt.subplot(132), plt.imshow(img), plt.title('lowpass_spatial')
# plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(f), plt.title('lowpass_frequency')
# plt.xticks([]), plt.yticks([])
# plt.show()





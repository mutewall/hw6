# -*- coding: UTF-8 -*-  
#廖沩健
#自动化65
#2160504124

from __future__ import division, print_function
import numpy as np
import random
import cv2
from scipy import stats
import math
import cmath

def load_file(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return img

def gaussian_noise(img, mean, std):
    noise = np.random.normal(mean, std, img.shape)
    noise = noise.reshape(img.shape)
    res = np.clip(img + noise, 0, 255)
    return res

def salt_pepper_noise(img, prob):
    res = np.zeros_like(img)
    threshold = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rnd = random.random()
            if rnd < prob:
                res[i,j] = 0
            elif rnd > threshold:
                res[i,j] = 255
            else:
                res[i,j] = img[i,j]
    
    return res

def arithmetic_mean(img):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            res[i-1,j-1] = np.mean(img[i-1:i+2,j-1:j+2])
    return res
 
def geometric_mean(img):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            patch = img[i-1:i+2,j-1:j+2]
            p = np.prod(patch.astype(np.float))
            res[i-1,j-1] = p ** (1 / (patch.shape[0] * patch.shape[1]))
    return res.astype(np.uint8)

def harmonic_mean(img):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            patch = img[i-1:i+2,j-1:j+2]
            patch = patch.astype(np.float)
            if 0 in patch:
                patch = 0
            else:
                patch = stats.hmean(patch.reshape(-1))
            res[i-1,j-1] = patch
    return res.astype(np.uint8)

def inverse_harmonic_mean(img, q):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            patch = img[i-1:i+2,j-1:j+2]
            patch = patch.astype(np.float)
            res[i-1, j-1] = np.mean(patch ** (q+1)) / np.mean(patch ** q)
    return res.astype(np.uint8)

def median_filter(img):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            res[i-1, j-1] = np.median(img[i-1:i+2,j-1:j+2])
    return res

def max_filter(img):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            res[i-1, j-1] = np.max(img[i-1:i+2,j-1:j+2])
    return res

def min_filter(img):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            res[i-1, j-1] = np.min(img[i-1:i+2,j-1:j+2])
    return res

def midpoint_filter(img):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            res[i-1, j-1] = 0.5*(np.max(img[i-1:i+2,j-1:j+2]) + np.min(img[i-1:i+2,j-1:j+2]))
    return res

def alpha_trimmed_mean_filter(img, d):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            patch = img[i-1:i+2,j-1:j+2].ravel()
            sorted_patch = np.sort(patch)
            res[i-1, j-1] = np.mean(sorted_patch[int(d/2):-int(d/2)])
    return res

def adaptive_filter(img, noise_std):
    res = np.zeros_like(img, dtype=np.float)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            patch = img[i-1:i+2,j-1:j+2]
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            res[i-1, j-1] = float(img[i,j]) - noise_std / patch_std * (float(img[i,j]) - patch_mean)
    res = np.clip(res, 0, 255)
    return res.astype(np.uint8)

def adaptive_median_filter(img):
    res = np.zeros_like(img)
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT, 0)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            patch = img[i-1:i+2,j-1:j+2]
            patch_max = np.max(patch)
            patch_min = np.min(patch)
            patch_med = np.median(patch)
            A1 = patch_med - patch_min
            A2 = patch_med - patch_max
            if A1 > 0 and A2 < 0:
                B1 = img[i,j] - patch_min
                B2 = img[i,j] - patch_max
                if B1 > 0 and B2 < 0:
                    res[i-1,j-1] = img[i,j]
                else:
                    res[i-1,j-1] = patch_med
            else:
                res[i-1, j-1] = patch_med
    return res

def blur_kernel(h, w, a, b, T):
    H = np.zeros((h,w), dtype=np.complex)
    for i in range(1, H.shape[0]+1):
        for j in range(1, H.shape[1]+1):
            H[i-1,j-1] = T / math.pi*(a*i + b*j) * math. \
                            sin(math.pi * (a * i + b * j)) \
                            * cmath.exp(-cmath.sqrt(-1) * math.pi * (a * i + b * j))
    return H

def frequency_blur(img, a, b, T):
    padded_img = np.pad(img, (0,img.shape[0]), 'constant')
    F = np.fft.fft2(padded_img)
    F = np.fft.fftshift(F)
    H = blur_kernel(2*img.shape[0], 2*img.shape[1], a, b, T)
    G = np.multiply(F, H)
    res = np.fft.ifftshift(G)
    res = np.abs(np.fft.ifft2(G))
    res = res[:img.shape[0], :img.shape[1]]
    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res.astype(np.uint8)

def wiener_restore(img, K, *args):
    a, b, T = args
    # padded_img = np.pad(img, (0,img.shape[0]), 'constant')
    # G = np.fft.fft2(padded_img)
    # G = np.fft.fftshift(G)
    G = np.fft.fft2(img)
    # H = blur_kernel(2*img.shape[0], 2*img.shape[1], a, b, T)
    H = blur_kernel(img.shape[0], img.shape[1], a, b, T)
    power_spectrum =  np.abs(H)**2
    # power_spectrum = np.vstack(map(lambda x: np.real(x)**2 + np.imag(x)**2, H))
    F_re = ((power_spectrum / (power_spectrum + K)) / H) * G
    # res = np.fft.ifftshift(F_re)
    # res = np.abs(np.fft.ifft2(res))
    # res = res[:img.shape[0], :img.shape[1]]
    res = np.real(np.fft.ifft2(F_re))
    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res.astype(np.uint8)
    # return res

def center_distance_matrix(img):
    centerX = img.shape[0] / 2
    centerY = img.shape[1] / 2
    cdm = np.zeros_like(img, dtype=np.float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cdm[i,j] = np.sqrt((i - centerX) ** 2 + (j - centerY) ** 2)
    return cdm

def constrained_least_square(img, gamma, *args):
    a, b, T = args
    G = np.fft.fft2(img)
    H = blur_kernel(img.shape[0], img.shape[1], a, b, T)
    power_spectrum = np.abs(H)**2
    power_spectrum
    cdm = center_distance_matrix(G)
    cdm = cdm.astype(np.complex)
    laplace_kernel = 1 + 4 * (math.pi ** 2) * (cdm ** 2)
    laplace_ps = np.abs(laplace_kernel) ** 2
    F_re = (np.conj(H) / (power_spectrum + gamma * laplace_ps)) * G
    res = np.real(np.fft.ifft2(F_re))
    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res.astype(np.uint8)



if __name__ == '__main__':
    img = load_file('lena.bmp')
    cv2.imwrite('./res/gaussian_{}_{}.bmp'.format(0,25), gaussian_noise(img, 0, 25))
    cv2.imwrite('./res/sp{}.bmp'.format(0.03), salt_pepper_noise(img, 0.03))
    cv2.imwrite('./res/am_sp.bmp', arithmetic_mean(salt_pepper_noise(img, 0.03)))
    cv2.imwrite('./res/am_g.bmp', arithmetic_mean(gaussian_noise(img, 0, 25)))
    cv2.imwrite('./res/gm_g.bmp', geometric_mean(gaussian_noise(img, 0, 25)))
    cv2.imwrite('./res/gm_sp.bmp', geometric_mean(salt_pepper_noise(img, 0.03)))
    cv2.imwrite('./res/hm_g.bmp', harmonic_mean(gaussian_noise(img, 0, 25)))
    cv2.imwrite('./res/hm_sp.bmp', harmonic_mean(salt_pepper_noise(img, 0.03)))
    cv2.imwrite('./res/ihm_g_{}.bmp'.format(0.2), inverse_harmonic_mean(gaussian_noise(img, 0, 25), 0.2))
    cv2.imwrite('./res/ihm_sp_{}.bmp'.format(0.2), inverse_harmonic_mean(salt_pepper_noise(img, 0.03), 0.2))
    cv2.imwrite('./res/med_g.bmp', median_filter(gaussian_noise(img, 0, 25)))
    cv2.imwrite('./res/med_sp.bmp', median_filter(salt_pepper_noise(img, 0.03)))
    cv2.imwrite('./res/max_g.bmp', max_filter(gaussian_noise(img, 0, 25)))
    cv2.imwrite('./res/max_sp.bmp', max_filter(salt_pepper_noise(img, 0.03)))
    cv2.imwrite('./res/min_g.bmp', min_filter(gaussian_noise(img, 0, 25)))
    cv2.imwrite('./res/min_sp.bmp', min_filter(salt_pepper_noise(img, 0.03)))
    cv2.imwrite('./res/mid_g.bmp', midpoint_filter(gaussian_noise(img, 0, 25)))
    cv2.imwrite('./res/mid_sp.bmp', midpoint_filter(salt_pepper_noise(img, 0.03)))
    cv2.imwrite('./res/alpha_g.bmp', alpha_trimmed_mean_filter(gaussian_noise(img, 0, 25), d=4))
    cv2.imwrite('./res/alpha_sp.bmp', alpha_trimmed_mean_filter(salt_pepper_noise(img, 0.03), d=4))
    cv2.imwrite('./res/ada_g.bmp', adaptive_filter(gaussian_noise(img, 0, 25), 25))
    cv2.imwrite('./res/ada_med_g.bmp', adaptive_median_filter(gaussian_noise(img, 0, 25)))
    cv2.imwrite('./res/ada_med_sp.bmp', adaptive_median_filter(salt_pepper_noise(img, 0.03)))
    cv2.imwrite('./res/blur.bmp', frequency_blur(img, 0.01, 0.01, 1))
    cv2.imwrite('./res/blur_gaussian.bmp', gaussian_noise(frequency_blur(img, 0.01, 0.01, 1), 0, 10))
    cv2.imwrite('./res/restore_blur_gaussian.bmp', wiener_restore(gaussian_noise(frequency_blur(img, 0.01, 0.01, 1), 0, 1), 1, 0.01, 0.01, 1))
    cv2.imwrite('./res/restore_blur_gaussian_cls.bmp', constrained_least_square(gaussian_noise(frequency_blur(img, 0.01, 0.01, 1), 0, 1), 1e-13, 0.01, 0.01, 1))








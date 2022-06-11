# -*- coding: utf-8 -*-
# ---
# @File: preprocessing.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/6/5
# Describe: 签名图像预处理
# 源：https://github.com/luizgh/sigver_wiwd
# ---

import cv2
import numpy as np
from scipy import ndimage


def hafemann_preprocess(img,ext_h,ext_w,dst_size=(150,220),img_size=(170, 242)):
    img=np.squeeze(img)
    centered_img=centered(img,ext_h,ext_w)  # 将图像的质心和幕布中心对齐
    inverted_img=255-centered_img  # 反色，背景为0，签名为255-x
    resized_img=resize_img(inverted_img,img_size)
    croped_img=crop_center(resized_img,dst_size)
    croped_img=croped_img.astype(np.uint8)
    return croped_img


def centered(img,ext_h,ext_w):
    # 先用高斯滤波去除图中的小组件
    radius=2
    blurred_img=ndimage.gaussian_filter(img,radius)
    # 求取质心
    threshold,binarized_img=cv2.threshold(blurred_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    r,c=np.where(binarized_img==0) # 有笔画的位置
    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())
    cropped = img[r.min(): r.max(), c.min(): c.max()] # 包围笔画的矩形框
    # 将笔画框的质心和幕布的中心对齐
    img_r, img_c = cropped.shape
    r_start = ext_h // 2 - r_center
    c_start = ext_w // 2 - c_center  # 求取对齐时笔画框左上角在幕布上的位置

    if img_r > ext_h: # 签名高度比幕布高度大
        print ('Warning: cropping image. The signature should be smaller than the canvas size')
        r_start = 0
        difference = img_r - ext_h
        crop_start = difference // 2
        cropped = cropped[crop_start:crop_start + ext_h, :]
        img_r = ext_h
    else:
        # 防止对齐后的笔画框超出幕布范围
        extra_r = (r_start + img_r) - ext_h
        if extra_r > 0:
            r_start -= extra_r
        if r_start < 0: # 如果要对齐左上角会超出幕布范围就放弃对齐
            r_start = 0

    if img_c > ext_w:
        print ('Warning: cropping image. The signature should be smaller than the canvas size')
        c_start = 0
        difference = img_c - ext_w
        crop_start = difference // 2
        cropped = cropped[:, crop_start:crop_start + ext_w]
        img_c = ext_w
    else:
        extra_c = (c_start + img_c) - ext_w
        if extra_c > 0:
            c_start -= extra_c
        if c_start < 0:
            c_start = 0
    normalized_image = np.ones((ext_h, ext_w), dtype=np.uint8) * 255
    normalized_image[r_start:r_start + img_r, c_start:c_start + img_c] = cropped
    normalized_image[normalized_image > threshold] = 255 # 用大津法确定的阈值去背景
    return normalized_image

def resize_img(img,new_size):
    # 先将一边缩小到指定大小，保持比例不变缩小另一边，剪切。
    dst_h,dst_w=new_size
    h_scale=float(img.shape[0])/dst_h
    w_scale=float(img.shape[1])/dst_w
    if w_scale>h_scale:
        resized_height=dst_h
        resized_width=int(round(img.shape[1]/h_scale))
    else:
        resized_width=dst_w
        resized_height=int(round(img.shape[0]/w_scale))
    img=cv2.resize(img.astype(np.float32),(resized_width,resized_height))
    if w_scale>h_scale:
        start = int(round((resized_width-dst_w)/2.0))
        return img[:, start:start+dst_w]
    else:
        start = int(round((resized_height-dst_h)/2.0))
        return img[start:start+dst_h, :]

def crop_center(img, input_shape):
    img_shape = img.shape
    start_y = (img_shape[0] - input_shape[0]) // 2
    start_x = (img_shape[1] - input_shape[1]) // 2
    cropped = img[start_y: start_y + input_shape[0], start_x:start_x + input_shape[1]]
    return cropped
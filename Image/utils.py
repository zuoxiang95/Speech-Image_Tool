# -*- coding: utf-8 -*-
"""
   author: 左想
   date: 2018-01-11
"""

import cv2
import random
import numpy as np
from math import fabs, sin, cos, radians
from PIL import Image, ImageDraw, ImageEnhance


def img_rotation(file_path, output, degree, is_full):
    """
    对图片进行旋转，并另存为旋转后的图片；
    :param file_path: String 图片路径；
    :param output: String 输出旋转后的图片路径；
    :param degree: String 旋转角度；
    :param is_full: Bool 是否保留整张图片进行旋转。
                    True则在旋转时会将尺寸进行扩大以保留完整的图片；
                    False则在旋转时保留原始图片的尺寸进行旋转；
    :return:
    """
    im = cv2.imread(file_path, 1)
    height, width = im.shape[:2]
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    if is_full:
        height_new = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        width_new = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        matRotation[0, 2] += (width_new - width) / 2  # 重点在这步，目前不懂为什么加这步
        matRotation[1, 2] += (height_new - height) / 2  # 重点在这步
        imgRotation = cv2.warpAffine(im, matRotation, (width_new, height_new), borderMode=cv2.BORDER_REPLICATE)
    else:
        imgRotation = cv2.warpAffine(im, matRotation, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return imgRotation


def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(0, 21) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(3, 15) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 15) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 21) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def randomGaussian(image, mean=0.5, sigma=0.3):
    """
     对图像进行高斯噪声处理
    :param image:
    :return:
    """

    def gaussianNoisy(im, mean=0.5, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    # 将图像转化成数组
    img = np.asarray(image)
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def translate_coord(left_top, length, width, theta, center=None):
    """
    根据旋转前矩形坐标以及旋转弧度来计算将矩形旋转弧度之后的顶点坐标
    :param left_top: 左下角顶点坐标
    :param length: 矩形长度
    :param width: 矩形宽度
    :param theta: 旋转弧度
    :return: 返回四个顶点坐标
    """
    # 获取左下角顶点坐标
    left_down = [left_top[0], left_top[1] + width]
    # 获取右上角顶点坐标
    right_top = [left_top[0] + length, left_top[1]]
    # 获取右下角顶点坐标
    right_down = [left_top[0] + length, left_top[1] + width]
    # 计算中心点坐标
    if center is None:
        center = [(left_top[0] + right_down[0]) / 2, (left_top[1] + right_down[1]) / 2]
    # 计算四个顶点旋转后的坐标
    right_down_rotation = calculate_rotation_coord(right_down, center, theta)
    right_top_rotation = calculate_rotation_coord(right_top, center, theta)
    left_down_rotation = calculate_rotation_coord(left_down, center, theta)
    left_top_rotation = calculate_rotation_coord(left_top, center, theta)
    return left_top_rotation, left_down_rotation, right_top_rotation, right_down_rotation


def calculate_rotation_coord(point, center, theta):
    """
    计算一个点以另一个点为中心，旋转theta弧度后的坐标
    :param point: 旋转前点的坐标
    :param center: 旋转中心坐标
    :param theta: 旋转弧度
    :return: 返回旋转后点的坐标
    """
    # 计算旋转之后点的坐标
    right_rotation_x = (point[0] - center[0]) * cos(theta) - \
                       (point[1] - center[1]) * sin(theta) + center[0]
    right_rotation_y = (point[0] - center[0]) * sin(theta) + \
                       (point[1] - center[1]) * cos(theta) + center[1]
    return [int(right_rotation_x), int(right_rotation_y)]


def draw_box(img, img_save, left_top, left_down, right_top, right_down):
    """
    根据矩形的四个点的坐标，在图片中画框
    :param img: 图片路径
    :param img_save: 图片保存路径
    :param left_top: 左上顶点坐标
    :param left_down: 左下顶点坐标
    :param right_top: 右上顶点坐标
    :param right_down: 右下顶点坐标
    :return: None
    """
    # 打开图片
    im = Image.open(img)
    draw = ImageDraw.Draw(im)
    # 分别画四条直线，即框出box的位置了
    draw.line((left_top[0], left_top[1], left_down[0], left_down[1]))
    draw.line((left_top[0], left_top[1], right_top[0], right_top[1]))
    draw.line((right_top[0], right_top[1], right_down[0], right_down[1]))
    draw.line((left_down[0], left_down[1], right_down[0], right_down[1]))

    im.save(img_save)


def save_image_use_cv2(image, path):
    cv2.imwrite(path, image)


def save_image_use_pil(image, path):
    image.save(path)

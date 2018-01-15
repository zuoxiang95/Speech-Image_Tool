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


def get_color_box(img, height_im, width_im, move_pix=10):
    """
    给定一个box的长宽以及移动的像素值，从图片的左上角开始移动，
    每次通过统计区域中像素颜色h的方差，找出图片h方差最小的区域也就是颜色相近的区域
    :param img: 样本图片
    :param height_im: 选取区域的长度
    :param width_im: 选取区域的宽度
    :param move_pix: 移动的像素值大小
    :return: 返回图片颜色相近区域的起始位置，该区域颜色的反色rgb值
    """
    im = cv2.imread(img)
    # 将rgb值转化为hsv值
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # rgb矩阵
    rgb_array = np.array(im)
    # hsv矩阵
    hsv_array = np.array(hsv_im)
    height, width, chanel = hsv_array.shape
    # 计算box需要移动的次数
    width_times = int((width - width_im) / move_pix + 1)
    height_times = int((height - height_im) / move_pix + 1)
    # 定义统计方差的list
    var_result = np.ndarray([height_times, width_times], dtype=np.float32)
    # 开始移动box
    for i in range(height_times):
        for j in range(width_times):
            # 计算box的起始位置
            begin_height = i * move_pix
            end_height = begin_height + height_im + 1
            begin_width = j * move_pix
            end_width = begin_width + width_im + 1
            # 获取到box对应的hsv数组
            hsv_box = hsv_array[begin_height:end_height, begin_width:end_width, :]
            # 计算box内的hsv中h的方差
            box_color_count = statistic_color(hsv_box)
            var_result[i, j] = box_color_count
    # 找出方差最小的box所在的行和列
    min_row, min_col = np.where(var_result == np.min(var_result))
    # 随机从符合的位置中选取一个
    rand_number = random.randint(0, len(min_row)-1)
    # 计算box对应的起始位置
    height_location_begin = min_row[rand_number] * move_pix
    height_location_end = height_location_begin + height_im
    width_location_begin = min_col[rand_number] * move_pix
    width_location_end = width_location_begin + width_im
    # 获取到box对应的rgb数组
    rgb_box = rgb_array[height_location_begin:height_location_end, width_location_begin:width_location_end, :]
    # 获取box的里的反色
    diff_max_rgb = get_diff_color(rgb_box)
    return [[height_location_begin, height_location_end], [width_location_begin, width_location_end]], diff_max_rgb


def statistic_color(color_array):
    """
    主要是获取box内的hsv值中h的方差
    :param color_array: box对应的矩阵
    :return: 返回box的hsv值中h的方差
    """
    h_value = color_array[:, :, 0]
    s_value = color_array[:, :, 1]
    variance = np.var(h_value) + np.var(s_value)
    return variance


def get_diff_color(color_array):
    """
    主要是获取当前box的rgb值的均值，再求均值的反色
    :param color_array: 当前box的rgb矩阵
    :return: 当前box的反色的rgb值
    """
    r_mean = np.mean(color_array[:, :, 0])
    g_mean = np.mean(color_array[:, :, 1])
    b_mean = np.mean(color_array[:, :, 2])
    return (int(255-r_mean), int(255-g_mean), int(255-b_mean))


def save_image_use_cv2(image, path):
    cv2.imwrite(path, image)


def save_image_use_pil(image, path):
    image.save(path)

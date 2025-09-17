import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import random


def parse_label_file(label_path):
    """解析YOLO格式的标签文件"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        objects = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                objects.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
        return objects
    except Exception as e:
        print(f"解析标签文件 {label_path} 出错: {str(e)}")
        return []


def write_label_file(label_path, objects):
    """写入YOLO格式的标签文件"""
    try:
        with open(label_path, 'w', encoding='utf-8') as f:
            for obj in objects:
                line = f"{obj['class_id']} {obj['x_center']:.6f} {obj['y_center']:.6f} {obj['width']:.6f} {obj['height']:.6f}\n"
                f.write(line)
    except Exception as e:
        print(f"写入标签文件 {label_path} 出错: {str(e)}")


def apply_blur_and_noise(img_path, label_path, augment_type='gaussian_blur', **kwargs):
    """
    应用模糊或噪声增强到图像
    :param img_path: 图像路径
    :param label_path: 标签路径
    :param augment_type: 增强类型 ('gaussian_blur', 'motion_blur', 'gaussian_noise', 'salt_pepper')
    :param kwargs: 增强参数
        gaussian_blur: sigma (模糊强度)
        motion_blur: k (核大小), angle (运动角度)
        gaussian_noise: scale (噪声强度)
        salt_pepper: prob (噪声概率)
    :return: 增强后的图像和原始标签对象列表 (标签不变)
    """
    try:
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return None, None

        # 读取标签 (噪声增强不改变标签)
        objects = parse_label_file(label_path)
        if not objects:
            print(f"标签文件为空或无效: {label_path}")
            return None, None

        # 创建增强器
        if augment_type == 'gaussian_blur':
            sigma = kwargs.get('sigma', 1.0)  # 默认模糊强度
            augmenter = iaa.GaussianBlur(sigma=sigma)
        elif augment_type == 'motion_blur':
            k = kwargs.get('k', 15)  # 核大小
            angle = kwargs.get('angle', 45)  # 运动角度
            augmenter = iaa.MotionBlur(k=k, angle=angle)
        elif augment_type == 'gaussian_noise':
            scale = kwargs.get('scale', 0.05)  # 噪声强度
            augmenter = iaa.AdditiveGaussianNoise(scale=scale*255)
        elif augment_type == 'salt_pepper':
            prob = kwargs.get('prob', 0.05)  # 噪声概率
            augmenter = iaa.SaltAndPepper(p=prob)
        else:
            print(f"不支持的增强类型: {augment_type}")
            return None, None

        # 应用增强 (标签不需要转换)
        img_aug = augmenter.augment_image(img)

        return img_aug, objects

    except Exception as e:
        print(f"应用 {augment_type} 增强到图像 {img_path} 时出错: {str(e)}")
        return None, None


def save_augmented_image_and_label(img_path, label_path, augment_type='gaussian_blur', **kwargs):
    """
    应用模糊或噪声增强并保存结果到当前目录
    :param img_path: 原始图像路径
    :param label_path: 原始标签路径
    :param augment_type: 增强类型 ('gaussian_blur', 'motion_blur', 'gaussian_noise', 'salt_pepper')
    :param kwargs: 增强参数
    """
    # 获取文件名和扩展名
    img_dir, img_filename = os.path.split(img_path)
    img_name, img_ext = os.path.splitext(img_filename)

    label_dir, label_filename = os.path.split(label_path)
    label_name, label_ext = os.path.splitext(label_filename)

    # 生成新文件名
    new_img_name = f"{img_name}_{augment_type}{img_ext}"
    new_label_name = f"{label_name}_{augment_type}{label_ext}"

    # 执行增强
    augmented_img, augmented_objects = apply_blur_and_noise(
        img_path, label_path, augment_type, **kwargs)

    if augmented_img is not None and augmented_objects is not None:
        # 保存增强后的图像
        cv2.imwrite(new_img_name, augmented_img)
        print(f"已保存增强图像: {new_img_name}")

        # 保存标签 (内容不变)
        write_label_file(new_label_name, augmented_objects)
        print(f"已保存增强标签: {new_label_name}")
    else:
        print("模糊/噪声增强失败，未保存文件")


# 使用示例
if __name__ == "__main__":
    # 输入图像和标签路径
    image_path = "example"  # 你的图像路径
    label_path = "example"  # 你的标签路径

    # 高斯模糊 (sigma=1.5)
    save_augmented_image_and_label(image_path, label_path,
                                 augment_type='gaussian_blur',
                                 sigma=1.5)

    # 运动模糊 (核大小=20, 角度=30度)
    save_augmented_image_and_label(image_path, label_path,
                                 augment_type='motion_blur',
                                 k=20, angle=30)

    # 高斯噪声 (强度=0.1)
    save_augmented_image_and_label(image_path, label_path,
                                 augment_type='gaussian_noise',
                                 scale=0.1)

    # 椒盐噪声 (概率=0.1)
    save_augmented_image_and_label(image_path, label_path,
                                 augment_type='salt_pepper',
                                 prob=0.1)
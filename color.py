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


def apply_color_adjustment(img_path, label_path, adjust_type='brightness', **kwargs):
    """
    应用颜色调整到图像
    :param img_path: 图像路径
    :param label_path: 标签路径
    :param adjust_type: 调整类型 ('brightness', 'contrast', 'hue_saturation', 'grayscale')
    :param kwargs: 调整参数
        brightness: factor (亮度系数，>1变亮，<1变暗)
        contrast: factor (对比度系数，>1增强，<1减弱)
        hue_saturation: hue (色调偏移量), saturation (饱和度系数)
    :return: 调整后的图像和原始标签对象列表 (标签不变)
    """
    try:
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return None, None

        # 读取标签 (颜色调整不改变标签)
        objects = parse_label_file(label_path)
        if not objects:
            print(f"标签文件为空或无效: {label_path}")
            return None, None

        # 创建增强器
        if adjust_type == 'brightness':
            factor = kwargs.get('factor', 1.5)  # 默认亮度系数
            augmenter = iaa.MultiplyBrightness(mul=factor)
        elif adjust_type == 'contrast':
            factor = kwargs.get('factor', 1.5)  # 默认对比度系数
            augmenter = iaa.LinearContrast(alpha=factor)
        elif adjust_type == 'hue_saturation':
            hue = kwargs.get('hue', 20)  # 默认色调偏移量(-180~180)
            saturation = kwargs.get('saturation', 1.5)  # 默认饱和度系数
            augmenter = iaa.MultiplyAndAddToBrightness(
                mul=saturation,
                add=hue,
                from_colorspace="BGR",
                to_colorspace="HSV"
            )
        elif adjust_type == 'grayscale':
            augmenter = iaa.Grayscale(alpha=1.0)
        else:
            print(f"不支持的颜色调整类型: {adjust_type}")
            return None, None

        # 应用增强 (标签不需要转换)
        img_aug = augmenter.augment_image(img)

        return img_aug, objects

    except Exception as e:
        print(f"应用 {adjust_type} 调整到图像 {img_path} 时出错: {str(e)}")
        return None, None


def save_adjusted_image_and_label(img_path, label_path, adjust_type='brightness', **kwargs):
    """
    应用颜色调整并保存结果到当前目录
    :param img_path: 原始图像路径
    :param label_path: 原始标签路径
    :param adjust_type: 调整类型 ('brightness', 'contrast', 'hue_saturation', 'grayscale')
    :param kwargs: 调整参数
    """
    # 获取文件名和扩展名
    img_dir, img_filename = os.path.split(img_path)
    img_name, img_ext = os.path.splitext(img_filename)

    label_dir, label_filename = os.path.split(label_path)
    label_name, label_ext = os.path.splitext(label_filename)

    # 生成新文件名
    new_img_name = f"{img_name}_{adjust_type}{img_ext}"
    new_label_name = f"{label_name}_{adjust_type}{label_ext}"

    # 执行调整
    adjusted_img, adjusted_objects = apply_color_adjustment(
        img_path, label_path, adjust_type, **kwargs)

    if adjusted_img is not None and adjusted_objects is not None:
        # 保存调整后的图像
        cv2.imwrite(new_img_name, adjusted_img)
        print(f"已保存调整图像: {new_img_name}")

        # 保存标签 (内容不变)
        write_label_file(new_label_name, adjusted_objects)
        print(f"已保存调整标签: {new_label_name}")
    else:
        print("颜色调整失败，未保存文件")


# 使用示例
if __name__ == "__main__":
    # 输入图像和标签路径
    image_path = "example"  # 你的图像路径
    label_path = "example"  # 你的标签路径

    # 亮度调整 (系数=1.8，变亮)
    save_adjusted_image_and_label(image_path, label_path,
                                adjust_type='brightness',
                                factor=1.8)

    # 对比度调整 (系数=2.0，增强对比度)
    save_adjusted_image_and_label(image_path, label_path,
                                adjust_type='contrast',
                                factor=2.0)

    # 色调饱和度调整 (色调偏移=30，饱和度=1.8)
    save_adjusted_image_and_label(image_path, label_path,
                                adjust_type='hue_saturation',
                                hue=30, saturation=1.8)

    # 灰度化
    save_adjusted_image_and_label(image_path, label_path,
                                adjust_type='grayscale')
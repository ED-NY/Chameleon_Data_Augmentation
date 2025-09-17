import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


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


def apply_geometric_transform(img_path, label_path, transform_type='rotate', **kwargs):
    """
    应用几何变换到图像和对应的标签
    :param img_path: 图像路径
    :param label_path: 标签路径
    :param transform_type: 变换类型 ('rotate', 'scale', 'translate', 'affine', 'elastic')
    :param kwargs: 变换参数
        rotate: angle (旋转角度)
        scale: scale_factor (缩放因子)
        translate: x_shift, y_shift (x,y平移量)
        affine: matrix (3x2仿射矩阵)
        elastic: alpha, sigma (弹性变换参数)
    :return: 变换后的图像和标签对象列表
    """
    try:
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return None, None

        # 读取标签
        objects = parse_label_file(label_path)
        if not objects:
            print(f"标签文件为空或无效: {label_path}")
            return None, None

        # 创建变换增强器
        if transform_type == 'rotate':
            angle = kwargs.get('angle', 30)
            augmenter = iaa.Affine(rotate=angle)
        elif transform_type == 'scale':
            scale = kwargs.get('scale_factor', 1.2)
            augmenter = iaa.Affine(scale=scale)
        elif transform_type == 'translate':
            x_shift = kwargs.get('x_shift', 0.1)
            y_shift = kwargs.get('y_shift', 0.1)
            augmenter = iaa.Affine(translate_percent={"x": x_shift, "y": y_shift})
        elif transform_type == 'affine':
            matrix = kwargs.get('matrix', np.array([[1, 0.5], [0, 1], [0.5, 0]]))
            augmenter = iaa.Affine(matrix=matrix)
        elif transform_type == 'elastic':
            alpha = kwargs.get('alpha', 10.0)
            sigma = kwargs.get('sigma', 5.0)
            augmenter = iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
        else:
            print(f"不支持的变换类型: {transform_type}")
            return None, None

        # 将边界框转换为imgaug格式
        bbs = []
        img_height, img_width = img.shape[:2]

        for obj in objects:
            x_center = obj['x_center']
            y_center = obj['y_center']
            width = obj['width']
            height = obj['height']

            # 转换为绝对坐标
            x1 = max(0, (x_center - width / 2) * img_width)
            y1 = max(0, (y_center - height / 2) * img_height)
            x2 = min(img_width, (x_center + width / 2) * img_width)
            y2 = min(img_height, (y_center + height / 2) * img_height)

            bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=obj['class_id']))

        # 创建BoundingBoxesOnImage对象
        bbs_oi = BoundingBoxesOnImage(bbs, shape=img.shape)

        # 应用变换
        seq_det = augmenter.to_deterministic()
        img_aug = seq_det.augment_image(img)
        bbs_aug = seq_det.augment_bounding_boxes([bbs_oi])[0]

        # 检查变换后的边界框是否在图像内
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        # 转换回相对坐标
        new_objects = []
        aug_height, aug_width = img_aug.shape[:2]

        for bb in bbs_aug.bounding_boxes:
            x_center = ((bb.x1 + bb.x2) / 2) / aug_width
            y_center = ((bb.y1 + bb.y2) / 2) / aug_height
            width = (bb.x2 - bb.x1) / aug_width
            height = (bb.y2 - bb.y1) / aug_height

            # 确保坐标在有效范围内
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            new_objects.append({
                'class_id': bb.label,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

        return img_aug, new_objects

    except Exception as e:
        print(f"应用几何变换 {transform_type} 到图像 {img_path} 时出错: {str(e)}")
        return None, None


def save_transformed_image_and_label(img_path, label_path, transform_type='rotate', **kwargs):
    """
    应用几何变换并保存结果到当前目录
    :param img_path: 原始图像路径
    :param label_path: 原始标签路径
    :param transform_type: 变换类型 ('rotate', 'scale', 'translate', 'affine', 'elastic')
    :param kwargs: 变换参数
    """
    # 获取文件名和扩展名
    img_dir, img_filename = os.path.split(img_path)
    img_name, img_ext = os.path.splitext(img_filename)

    label_dir, label_filename = os.path.split(label_path)
    label_name, label_ext = os.path.splitext(label_filename)

    # 生成新文件名
    new_img_name = f"{img_name}_{transform_type}{img_ext}"
    new_label_name = f"{label_name}_{transform_type}{label_ext}"

    # 执行变换
    transformed_img, transformed_objects = apply_geometric_transform(
        img_path, label_path, transform_type, **kwargs)

    if transformed_img is not None and transformed_objects is not None:
        # 保存变换后的图像
        cv2.imwrite(new_img_name, transformed_img)
        print(f"已保存变换图像: {new_img_name}")

        # 保存变换后的标签
        write_label_file(new_label_name, transformed_objects)
        print(f"已保存变换标签: {new_label_name}")
    else:
        print("几何变换失败，未保存文件")


# 使用示例
if __name__ == "__main__":
    # 输入图像和标签路径
    image_path = "example"  # 你的图像路径
    label_path = "example"  # 你的标签路径

    # 旋转变换 (30度)
    save_transformed_image_and_label(image_path, label_path,
                                   transform_type='rotate', angle=30)

    # 缩放变换 (1.5倍)
    save_transformed_image_and_label(image_path, label_path,
                                   transform_type='scale', scale_factor=1.5)

    # 平移变换 (x方向10%, y方向20%)
    save_transformed_image_and_label(image_path, label_path,
                                   transform_type='translate',
                                   x_shift=0.1, y_shift=0.2)

    # 仿射变换 (自定义3x2矩阵)
    affine_matrix = np.array([[1, 0.3], [0.2, 1], [0.1, 0.1]])
    save_transformed_image_and_label(image_path, label_path,
                                   transform_type='affine',
                                   matrix=affine_matrix)

    # 弹性变换 (alpha=15, sigma=5)
    save_transformed_image_and_label(image_path, label_path,
                                   transform_type='elastic',
                                   alpha=15.0, sigma=5.0)
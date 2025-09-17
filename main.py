import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import random
import argparse


# 共享的函数定义
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


# 颜色调整增强
def apply_color_adjustment(img, objects, adjust_type='brightness', **kwargs):
    """
    应用颜色调整到图像
    :param img: 输入图像
    :param objects: 标签对象列表
    :param adjust_type: 调整类型 ('brightness', 'contrast', 'hue_saturation', 'grayscale')
    :param kwargs: 调整参数
        brightness: factor (亮度系数，>1变亮，<1变暗)
        contrast: factor (对比度系数，>1增强，<1减弱)
        hue_saturation: hue (色调偏移量), saturation (饱和度系数)
    :return: 调整后的图像和原始标签对象列表 (标签不变)
    """
    try:
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
        print(f"应用 {adjust_type} 调整时出错: {str(e)}")
        return None, None


# 翻转增强
def flip_image_and_labels(img, objects, flip_type='horizontal'):
    """
    翻转图像和对应的标签
    :param img: 输入图像
    :param objects: 标签对象列表
    :param flip_type: 翻转类型 ('horizontal'或'vertical')
    :return: 翻转后的图像和标签对象列表
    """
    try:
        # 创建翻转增强器
        if flip_type == 'horizontal':
            augmenter = iaa.Fliplr(1.0)  # 100%水平翻转
        elif flip_type == 'vertical':
            augmenter = iaa.Flipud(1.0)  # 100%垂直翻转
        else:
            print(f"不支持的翻转类型: {flip_type}")
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

        # 应用翻转
        seq_det = augmenter.to_deterministic()
        img_aug = seq_det.augment_image(img)
        bbs_aug = seq_det.augment_bounding_boxes([bbs_oi])[0]

        # 检查翻转后的边界框是否在图像内
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
        print(f"翻转图像时出错: {str(e)}")
        return None, None


# 几何变换增强
def apply_geometric_transform(img, objects, transform_type='rotate', **kwargs):
    """
    应用几何变换到图像和对应的标签
    :param img: 输入图像
    :param objects: 标签对象列表
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
        print(f"应用几何变换 {transform_type} 时出错: {str(e)}")
        return None, None


# 模糊和噪声增强
def apply_blur_and_noise(img, objects, augment_type='gaussian_blur', **kwargs):
    """
    应用模糊或噪声增强到图像
    :param img: 输入图像
    :param objects: 标签对象列表
    :param augment_type: 增强类型 ('gaussian_blur', 'motion_blur', 'gaussian_noise', 'salt_pepper')
    :param kwargs: 增强参数
        gaussian_blur: sigma (模糊强度)
        motion_blur: k (核大小), angle (运动角度)
        gaussian_noise: scale (噪声强度)
        salt_pepper: prob (噪声概率)
    :return: 增强后的图像和原始标签对象列表 (标签不变)
    """
    try:
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
            augmenter = iaa.AdditiveGaussianNoise(scale=scale * 255)
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
        print(f"应用 {augment_type} 增强时出错: {str(e)}")
        return None, None


# 主处理函数
def process_image_and_label(img_path, label_path, output_dir, augment_type, **kwargs):
    """
    处理单个图像和标签文件
    :param img_path: 图像路径
    :param label_path: 标签路径
    :param output_dir: 输出目录
    :param augment_type: 增强类型
    :param kwargs: 增强参数
    """
    try:
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return

        # 读取标签
        objects = parse_label_file(label_path)
        if not objects:
            print(f"标签文件为空或无效: {label_path}")
            return

        # 获取文件名和扩展名
        img_filename = os.path.basename(img_path)
        img_name, img_ext = os.path.splitext(img_filename)
        label_filename = os.path.basename(label_path)
        label_name, label_ext = os.path.splitext(label_filename)

        # 根据增强类型调用不同的处理函数
        if augment_type in ['brightness', 'contrast', 'hue_saturation', 'grayscale']:
            img_aug, objects_aug = apply_color_adjustment(img, objects, augment_type, **kwargs)
            suffix = augment_type
        elif augment_type in ['horizontal_flip', 'vertical_flip']:
            flip_type = augment_type.split('_')[0]
            img_aug, objects_aug = flip_image_and_labels(img, objects, flip_type)
            suffix = f"{flip_type}_flip"
        elif augment_type in ['rotate', 'scale', 'translate', 'affine', 'elastic']:
            img_aug, objects_aug = apply_geometric_transform(img, objects, augment_type, **kwargs)
            suffix = augment_type
        elif augment_type in ['gaussian_blur', 'motion_blur', 'gaussian_noise', 'salt_pepper']:
            img_aug, objects_aug = apply_blur_and_noise(img, objects, augment_type, **kwargs)
            suffix = augment_type
        else:
            print(f"不支持的增强类型: {augment_type}")
            return

        if img_aug is None or objects_aug is None:
            print(f"增强处理失败: {img_path}")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成新文件名
        new_img_name = f"{img_name}_{suffix}{img_ext}"
        new_label_name = f"{label_name}_{suffix}{label_ext}"
        new_img_path = os.path.join(output_dir, new_img_name)
        new_label_path = os.path.join(output_dir, new_label_name)

        # 保存增强后的图像和标签
        cv2.imwrite(new_img_path, img_aug)
        write_label_file(new_label_path, objects_aug)
        print(f"已保存增强结果: {new_img_path} 和 {new_label_path}")

    except Exception as e:
        print(f"处理 {img_path} 时出错: {str(e)}")


# 批量处理函数
def batch_process(input_dir, output_dir, augment_type, **kwargs):
    """
    批量处理目录中的图像和标签文件
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param augment_type: 增强类型
    :param kwargs: 增强参数
    """
    # 获取所有图像文件
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [f for f in os.listdir(input_dir)
                 if os.path.splitext(f)[1].lower() in img_extensions]

    # 处理每个图像文件
    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)
        img_name = os.path.splitext(img_file)[0]

        # 查找对应的标签文件
        label_extensions = ['.txt']
        label_file = None
        for ext in label_extensions:
            possible_label = img_name + ext
            if os.path.exists(os.path.join(input_dir, possible_label)):
                label_file = possible_label
                break

        if label_file:
            label_path = os.path.join(input_dir, label_file)
            process_image_and_label(img_path, label_path, output_dir, augment_type, **kwargs)
        else:
            print(f"未找到 {img_file} 的对应标签文件")


# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='YOLO 数据增强工具')
    parser.add_argument('input', help='输入目录或文件路径')
    parser.add_argument('output', help='输出目录路径')
    parser.add_argument('--augment', required=True,
                        choices=['brightness', 'contrast', 'hue_saturation', 'grayscale',
                                 'horizontal_flip', 'vertical_flip',
                                 'rotate', 'scale', 'translate', 'affine', 'elastic',
                                 'gaussian_blur', 'motion_blur', 'gaussian_noise', 'salt_pepper'],
                        help='选择增强类型')

    # 颜色调整参数
    parser.add_argument('--brightness_factor', type=float, help='亮度调整系数')
    parser.add_argument('--contrast_factor', type=float, help='对比度调整系数')
    parser.add_argument('--hue_shift', type=int, help='色调偏移量')
    parser.add_argument('--saturation_factor', type=float, help='饱和度调整系数')

    # 几何变换参数
    parser.add_argument('--rotate_angle', type=float, help='旋转角度')
    parser.add_argument('--scale_factor', type=float, help='缩放因子')
    parser.add_argument('--x_shift', type=float, help='x方向平移量')
    parser.add_argument('--y_shift', type=float, help='y方向平移量')
    parser.add_argument('--elastic_alpha', type=float, help='弹性变换alpha参数')
    parser.add_argument('--elastic_sigma', type=float, help='弹性变换sigma参数')

    # 模糊和噪声参数
    parser.add_argument('--blur_sigma', type=float, help='高斯模糊sigma参数')
    parser.add_argument('--motion_k', type=int, help='运动模糊核大小')
    parser.add_argument('--motion_angle', type=int, help='运动模糊角度')
    parser.add_argument('--noise_scale', type=float, help='高斯噪声强度')
    parser.add_argument('--salt_pepper_prob', type=float, help='椒盐噪声概率')

    return parser.parse_args()


# 主函数
def main():
    args = parse_args()

    # 准备增强参数
    augment_params = {}

    if args.augment == 'brightness' and args.brightness_factor:
        augment_params['factor'] = args.brightness_factor
    elif args.augment == 'contrast' and args.contrast_factor:
        augment_params['factor'] = args.contrast_factor
    elif args.augment == 'hue_saturation':
        if args.hue_shift:
            augment_params['hue'] = args.hue_shift
        if args.saturation_factor:
            augment_params['saturation'] = args.saturation_factor
    elif args.augment == 'rotate' and args.rotate_angle:
        augment_params['angle'] = args.rotate_angle
    elif args.augment == 'scale' and args.scale_factor:
        augment_params['scale_factor'] = args.scale_factor
    elif args.augment == 'translate':
        if args.x_shift:
            augment_params['x_shift'] = args.x_shift
        if args.y_shift:
            augment_params['y_shift'] = args.y_shift
    elif args.augment == 'elastic':
        if args.elastic_alpha:
            augment_params['alpha'] = args.elastic_alpha
        if args.elastic_sigma:
            augment_params['sigma'] = args.elastic_sigma
    elif args.augment == 'gaussian_blur' and args.blur_sigma:
        augment_params['sigma'] = args.blur_sigma
    elif args.augment == 'motion_blur':
        if args.motion_k:
            augment_params['k'] = args.motion_k
        if args.motion_angle:
            augment_params['angle'] = args.motion_angle
    elif args.augment == 'gaussian_noise' and args.noise_scale:
        augment_params['scale'] = args.noise_scale
    elif args.augment == 'salt_pepper' and args.salt_pepper_prob:
        augment_params['prob'] = args.salt_pepper_prob

    # 处理输入路径
    if os.path.isfile(args.input):
        # 处理单个文件
        img_path = args.input
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(os.path.dirname(img_path), f"{img_name}.txt")

        if not os.path.exists(label_path):
            print(f"未找到对应的标签文件: {label_path}")
            return

        process_image_and_label(img_path, label_path, args.output, args.augment, **augment_params)
    else:
        # 处理目录
        batch_process(args.input, args.output, args.augment, **augment_params)


if __name__ == "__main__":
    main()
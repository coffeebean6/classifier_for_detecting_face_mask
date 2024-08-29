import random
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms

# 自定义transform方法
class RandomFaceCutout(object):
    """Randomly mask out one patche from an face image.
    Args:
        p (float): probability.
    """
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image):
        """
        Args:
            image: PIL.Image
        Returns:
            PIL.Image
        """
        if random.uniform(0, 1) > self.p:
            return image
        
        # 人脸关键点检测
        face_landmarks_list = face_recognition.face_landmarks(np.array(image))
        # 无人脸返回
        if (len(face_landmarks_list)) <= 0:
            return image
        
        # 只处理第一张脸
        face_landmarks = face_landmarks_list[0]
        mask_numbers = (0, 1, 3, 4) # 0遮眼、1遮鼻、2遮嘴、3遮左脸、2遮右脸
        random_mask = random.choice(mask_numbers)
        if random_mask == 0: # 0遮眼
            # 取关键点坐标
            left_eyebrow = face_landmarks['left_eyebrow']
            right_eyebrow = face_landmarks['right_eyebrow']
            left_eye = face_landmarks['left_eye']
            nose_bridge = face_landmarks['nose_bridge']
            # 确定长方形的四个点
            left = left_eyebrow[0][0]
            top = left_eyebrow[2][1]
            right = right_eyebrow[4][0]
            bottom = nose_bridge[1][1]
            # 绘制矩形遮罩
            draw = ImageDraw.Draw(image)
            draw.rectangle([left, top, right, bottom], fill="black")
        if random_mask == 1: # 1遮鼻
            nose_bridge = face_landmarks['nose_bridge']
            nose_tip = face_landmarks['nose_tip']
            # 确定长方形的四个点
            left = nose_tip[0][0]
            top = nose_bridge[0][1]
            right = nose_tip[4][0]
            bottom = nose_tip[0][1]
            # 绘制矩形遮罩
            draw = ImageDraw.Draw(image)
            draw.rectangle([left, top, right, bottom], fill="black")
        if random_mask == 2: # 2遮嘴
            top_lip = face_landmarks['top_lip']
            chin = face_landmarks['chin']
            # 确定点
            points = [chin[2], top_lip[1], top_lip[4], chin[14], chin[11], chin[8], chin[5]]
            # 绘制遮罩
            draw = ImageDraw.Draw(image)
            draw.polygon(points, fill='black')
        if random_mask == 3: # 3遮左半脸
            left_eyebrow = face_landmarks['left_eyebrow']
            nose_bridge = face_landmarks['nose_bridge']
            chin = face_landmarks['chin']
            # 确定点
            points = [left_eyebrow[0], left_eyebrow[2], left_eyebrow[4], nose_bridge[0], chin[8], chin[4]]
            # 绘制遮罩
            draw = ImageDraw.Draw(image)
            draw.polygon(points, fill='black')
        if random_mask == 4: # 4遮右半脸
            right_eyebrow = face_landmarks['right_eyebrow']
            nose_bridge = face_landmarks['nose_bridge']
            chin = face_landmarks['chin']
            # 确定点
            points = [right_eyebrow[0], right_eyebrow[2], right_eyebrow[4], chin[14], chin[11], chin[8], nose_bridge[0]]
            # 绘制遮罩
            draw = ImageDraw.Draw(image)
            draw.polygon(points, fill='black')

        return image


# 在训练部分被引用
# 传入两个参数img_和train_transform，返回PIL image，也就是可以直接plot将其格式化
def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    # 对normalize进行反操作
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize),
                                     transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean,
                            dtype=img_.dtype,
                            device=img_.device)
        std = torch.tensor(norm_transform[0].std,
                           dtype=img_.dtype,
                           device=img_.device)
        # normalize是减去均值除于方差，因此反操作就是乘于方差再加上均值
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])
 
    # 通道变换，C*H*W --> H*W*C，将channel放到最后面
    img_ = img_.transpose(0, 2).transpose(0, 1)
    
    # 将0-1尺度上的数据转换到0-255
    if 'ToTensor' in str(transform_train) or img_.max() < 1:
        img_ = img_.detach().numpy() * 255
 
    # 将np_array的形式转换成PIL image
    # 判断channel是3通道还是1通道，分别转换成RGB彩色图像和灰度图像
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!"
                        .format(img_.shape[2]) )
    # 返回图像就可以对图像进行plot，对图像进行可视化
    return img_
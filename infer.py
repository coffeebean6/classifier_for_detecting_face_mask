import numpy as np
from PIL import Image
import os
import timm
import torch
import torchvision.transforms as transforms
from multiprocessing import cpu_count


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f' **** Device Type: {device}')
if torch.cuda.device_count() > 0:
    cuda_device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f' **** Device Name: {cuda_device_names}')
os_cpu_cores = os.cpu_count()
cpu_cores = cpu_count()
print(f" **** CPU Cores: {os_cpu_cores}/{cpu_cores}")
print(f' **** Torch Version: {torch.__version__}')


class CFG:
    base_model = "resnet18d.ra2_in1k"
    num_classes = 2
    state_dict_path = 'model-1.0_e4-92.12.pt'


model = timm.create_model(CFG.base_model, pretrained=False, num_classes=CFG.num_classes)
state_dict = torch.load(CFG.state_dict_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.to(device)
print(f" **** Model loaded from: {CFG.state_dict_path}")


bicubic = Image.BICUBIC

# 定义图像预处理步骤  
preprocess = transforms.Compose([
    transforms.Resize(size=256, interpolation=bicubic, max_size=None, antialias=True),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
])

# 加载单张图像
def infer_pic(img_path):
    input_image = Image.open(img_path).convert('RGB')  
    input_tensor = preprocess(input_image)  
    input_batch = input_tensor.unsqueeze(0)  # 添加批次维度  
    input_batch = input_batch.to(device)

    with torch.no_grad():  
        output = model(input_batch)
        
    # 对output取softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(f' **** Single Test: Softmaxed probabilities: {probabilities}')
    predicted_class = torch.argmax(probabilities).item()  
    print(f' **** Single Test: Predicted class: {predicted_class}')
    return predicted_class

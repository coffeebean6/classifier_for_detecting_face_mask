import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import timm
import time
import math
import os
from multiprocessing import cpu_count
import random
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from metrics import AverageMeter, ProgressMeter
from transform import RandomFaceCutout, transform_invert


def set_seeds(seed):
    """Set seeds for reproducibility """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
set_seeds(seed=2024)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f' **** Device Type: {device}')
if torch.cuda.device_count() > 0:
    cuda_device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f' **** Device Name: {cuda_device_names}')
os_cpu_cores = os.cpu_count()
cpu_cores = cpu_count()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(f" **** CPU Cores: {os_cpu_cores}/{cpu_cores}")
print(f' **** Torch Version: {torch.__version__}')


train_label = pd.read_csv('data/mask/train.csv')
val_label = pd.read_csv('data/mask/valid.csv')
print(f' **** Train Dataset Info: {train_label.info()}')
print(f' **** Valid Dataset Info: {val_label.info()}')


class CFG:
    tag = 1.0 #训练tag
    num_classes = 2
    num_epochs = 5
    batch_size = 32
    num_steps = math.ceil(len(train_label) / batch_size)
    num_workers = 0 #os.cpu_count() // 2
    max_lr = 4e-5
    drop_rate = 0.35
    weight_decay = 0.2
    ema_alpha = 0.99
    print_steps = 10

writer = SummaryWriter(f'./logs/{CFG.tag}')

# 构建数据集
class LiamDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(np.array(self.img_label[index]))
    
    def __len__(self):
        return len(self.img_path)
    

bicubic = Image.BICUBIC
train_transform = transforms.Compose([
    transforms.Resize(size=256, interpolation=bicubic, max_size=None, antialias=True),
    RandomFaceCutout(p=0.7), # 70% 加人脸mask cutout
    transforms.RandomRotation(degrees=30, expand=True),
    transforms.CenterCrop(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
])

train_loader = torch.utils.data.DataLoader(
    LiamDataset(
        train_label['path'], 
        train_label['target'],
        train_transform
    ), batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    LiamDataset(
        val_label['path'], 
        val_label['target'], 
        transforms.Compose([
            transforms.Resize(size=256, interpolation=bicubic, max_size=None, antialias=True),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
        ])
    ), batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True
)


def update_ema(ema_model, model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data * (1 - alpha))


# 模型训练
def train(train_loader, model, ema_model, criterion, optimizer, epoch, val_loader):
    batch_time = AverageMeter('Time', ':6.3f') # 每个epoch会重新做平均
    metrics_loss = AverageMeter('Loss', ':.4e')
    metrics_acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, metrics_loss, metrics_acc)

    # switch to train mode
    model.train()
    
    curr_best_acc = 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)

        # img_tensor = input[0, ...].cpu() # C H W
        # img = transform_invert(img_tensor, train_transform)
        # plt.imshow(img)
        # plt.show()
        # if i >= 10:
        #     break
        
        # compute output
        output = model(input)
        loss = criterion(output, target)
        acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() #for OneCycle cosine LR
        
        # update EMA parameters
        update_ema(ema_model, model, CFG.ema_alpha)
        
        # record metrics
        metrics_loss.update(loss.item(), input.size(0))
        metrics_acc.update(acc, input.size(0))
        step = epoch * CFG.num_steps + i
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], step)
        writer.add_scalar("Loss", loss.item() / input.size(0), step)
        writer.add_scalar("Acc", acc, step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % CFG.print_steps == 0:
            progress.pr2int(i)

        if i % 5000 == 0 and i > 0:
            val_acc = validate(val_loader, ema_model, criterion)
            print(f" **** Valid Accuracy: {val_acc.avg.item()}")
            if val_acc.avg.item() > curr_best_acc: # 保存
                curr_best_acc = round(val_acc.avg.item(), 2)
                torch.save(ema_model.state_dict(), f'./model-{CFG.tag}_e{epoch}-{i}_{curr_best_acc}.pt')
                print(f" **** Saved Checkpoint: ./model-{CFG.tag}_e{epoch}-{i}_{curr_best_acc}.pt")
            model.train() # 恢复成 train mode

            
# 模型验证
def validate(val_loader, ema_model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    metrics_loss = AverageMeter('Loss', ':.4e')
    metrics_acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, metrics_loss, metrics_acc)

    # switch to evaluate mode
    ema_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = ema_model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
            metrics_loss.update(loss.item(), input.size(0))
            metrics_acc.update(acc, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % CFG.print_steps == 0:
                progress.pr2int(i)

        # TODO: this should also be done with the ProgressMeter
        print(f' **** Valid Acc {metrics_acc.avg:.3f}')
        return metrics_acc


#基模型
model_name = "resnet18d.ra2_in1k"
model = timm.create_model(
    model_name, 
    pretrained=True, 
    num_classes=CFG.num_classes,
    drop_rate=CFG.drop_rate
)
#加载checkpoint，继续训练
# state_dict_path = 'model-2-30849_95.98.pt'
# state_dict = torch.load(state_dict_path)
# model.load_state_dict(state_dict, strict=False)
model = model.to(device)
ema_model = timm.create_model(model_name, pretrained=True, num_classes=CFG.num_classes, drop_rate=CFG.drop_rate)
ema_model = ema_model.to(device)
ema_model.load_state_dict(model.state_dict())

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = DataParallel(model)  # 自动复制模型到所有GPU

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=CFG.max_lr, 
    betas=(0.9, 0.999), 
    eps=1e-8, 
    weight_decay=CFG.weight_decay
)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
scheduler = OneCycleLR(
    optimizer, 
    max_lr=CFG.max_lr, 
    steps_per_epoch=CFG.num_steps, 
    epochs=CFG.num_epochs,
    div_factor=20, #初始lr, 默认25，不宜过小
    final_div_factor=1000, #最后lr会除以这个因子，默认值是10000.0，不宜过小
    anneal_strategy='cos'
)

best_acc = 0.0
for epoch in range(CFG.num_epochs):
    #scheduler.step() # for stepRL
    print(' **** Epoch: ', epoch)

    train(train_loader, model, ema_model, criterion, optimizer, epoch, val_loader)
    val_acc = validate(val_loader, ema_model, criterion)
    
    if val_acc.avg.item() > best_acc:
        best_acc = round(val_acc.avg.item(), 2)
        torch.save(ema_model.state_dict(), f'./model-{CFG.tag}_e{epoch}-{best_acc}.pt')

writer.close()


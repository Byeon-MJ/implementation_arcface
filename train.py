import torch
print(torch.__version__)

#라이브러리 세팅
import random
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as model

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split

# GPU 확인
print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

torch.backends.cudnn.enabled = False

CFG = {
    'IMG_SIZE':456,
    'EPOCHS':20,
    'LEARNING_RATE':0.000003,
    'BATCH_SIZE':16,
    'SEED':42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(CFG['SEED'])


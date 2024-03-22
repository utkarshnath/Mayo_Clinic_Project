import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d
import csv
import numpy as np
import nibabel as nib
import cv2
import torchvision.transforms as transforms
from torchsummary import summary
import sys
from utils import *
from config import models_genesis_config
from tqdm import tqdm
from PIL import Image
import argparse
from dataset import Y90Dataset, Y90WeightedDataset, Y90PrePostDataset
from models import TargetNet, MultiHeadNet, JointHeadNet, WeightedUnet3D, SiameseWeightedUnet3D, SiameseWeightedUnet3DGradCam
from classification import get_patient_dict, CosineScheduler, loadEncoder
from losses import ContrastiveLoss
from medcam import medcam

conf = models_genesis_config()
conf.display()

patient_dict = get_patient_dict()
series = 'data'

dataset_val = Y90PrePostDataset('/scratch/unath/Y90-Project/ExtractedFromISD_SILVA_HCC_Y90_1', '/scratch/unath/Y90-Project/patientKey/test_data_top.csv', patient_dict, train_type=1)
val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)

base_model = unet3d.UNet3DClassification()
base_model = loadEncoder(base_model, 'pretrained_weights/Y90Genesis_Chest_CT.pt')

target_model = SiameseWeightedUnet3DGradCam(base_model, n_class=1)

# For testing
path = "/data/yyang409/unath/pretrained_weights/classification/SiameseWeighteUnet3D-top-data/50.pt"
checkpoint = torch.load(path)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
target_model.load_state_dict(unParalled_state_dict)

device = 'cuda'
target_model.to(device)
#target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])

target_model = medcam.inject(target_model, output_dir="attention_maps", layer=['base_model.down_tr64', 'base_model.down_tr128', 'base_model.down_tr256', 'base_model.down_tr512'],save_maps=True)

correct = 0
total = 0
with torch.no_grad():
    target_model.eval()
    print("validating....")
    for batch_ndx, (x_pre, x_post, y) in enumerate(val_loader):
        x_pre, x_post, y = x_pre.float().to(device), x_post.float().to(device), y.float().to(device)
        pre_out = target_model(x_post)
        '''
        diff = pre_out - post_out
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        print(dist, y)
        predicted = dist > 0.1
        total += y.size(0)
        correct += (predicted == y.to(device)).sum()
        '''
    #print('Accuracy: %.2f %%' % (100 * float(correct) / total))

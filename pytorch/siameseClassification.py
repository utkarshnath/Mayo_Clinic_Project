import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from uunet import NestedUNet
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
from dataset import Y90Dataset, Y90WeightedDataset, Y90PrePostDataset, Y90PrePostSegDataset
from models import TargetNet, MultiHeadNet, JointHeadNet, WeightedUnet3D, SiameseWeightedUnet3D
from classification import get_patient_dict, CosineScheduler, loadEncoder
from losses import ContrastiveLoss

conf = models_genesis_config()
conf.display()

patient_dict = get_patient_dict()
series = 'data'

def train(train_loader, val_loader, model, optimizer, scheduler, criterion):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    best_loss = 100000
    intial_epoch = 0
    nb_epoch = 100
    num_epoch_no_improvement = 0
    sys.stdout.flush()
    temperature = 1.0

    for epoch in range(intial_epoch, nb_epoch):
        scheduler.step(epoch)
        target_model.train()
        for batch_ndx, (x_pre, x_post, y) in enumerate(train_loader):
            x_pre, x_post, y = x_pre.float().to(device), x_post.float().to(device), y.float().to(device)
            pre_out, post_out = target_model(x_pre, x_post)
            
            loss = criterion(pre_out, post_out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))
            torch.save({
              'epoch': epoch+1,
              'state_dict' : target_model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(conf.model_path, str(epoch)+".pt"))

            if (batch_ndx + 1) % 3 ==0:
               print('Epoch [{}/{}], batch_ndx {}, Loss: {:.6f}'.format(epoch + 1, conf.nb_epoch, batch_ndx + 1, np.average(train_losses)))
               sys.stdout.flush()

        correct = 0
        total = 0
        with torch.no_grad():
             target_model.eval()
             print("validating....")
             for batch_ndx, (x_pre, x_post, y) in enumerate(val_loader):
                 x_pre, x_post, y = x_pre.float().to(device), x_post.float().to(device), y.float().to(device)
                 pre_out, post_out = target_model(x_pre, x_post)
                 loss = criterion(pre_out, post_out, y)
                 valid_losses.append(loss.item())
 
                 diff = pre_out - post_out
                 dist_sq = torch.sum(torch.pow(diff, 2), 1)
                 dist = torch.sqrt(dist_sq)
                 print(dist, y)

            
                 #predicted = torch.round(pred[:,0])

                 #total += y.size(0)
                 #correct += (predicted == y.to(device)).sum()
             #print('Accuracy: %.2f %%' % (100 * float(correct) / total))

if __name__ == "__main__":
  dataset_train = Y90PrePostSegDataset('/scratch/unath/Y90-Project/ExtractedFromISD_SILVA_HCC_Y90_1', '/scratch/unath/Y90-Project/patientKey/train_data_top.csv', patient_dict, train_type=0)
  train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True)

  dataset_val = Y90PrePostSegDataset('/scratch/unath/Y90-Project/ExtractedFromISD_SILVA_HCC_Y90_1', '/scratch/unath/Y90-Project/patientKey/test_data_top.csv', patient_dict, train_type=1)
  val_loader = DataLoader(dataset_val, batch_size=4, shuffle=False)

  # base_model = unet3d.UNet3DClassification()
  base_model = unet3d.UNet3DClassificationDense()
  base_model = loadEncoder(base_model, 'pretrained_weights/Y90Genesis_Chest_CT.pt')

  target_model = SiameseWeightedUnet3D(base_model, n_class=1)

  device = 'cuda'
  target_model.to(device)
  target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])

  criterion = ContrastiveLoss()
  optimizer = torch.optim.SGD(target_model.parameters(), 0.1, momentum=0.9, weight_decay=0.0, nesterov=False)

  scheduler_func = CosineScheduler(100, 0.1, 1e-4)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_func) 

  train(train_loader, val_loader, target_model, optimizer, scheduler, criterion)

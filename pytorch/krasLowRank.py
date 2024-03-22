import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d
import denseUnet3d
from uunet import NestedUNet
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
from dataset import KrasClassificationDataset, KrasClassificationDatasetV2
from models import TargetNet, TargetNetV2, Classifier, LowRankLinear
from losses import ContrastiveLoss

conf = models_genesis_config()
conf.display()

class CosineScheduler:
    def __init__(self, steps, base_lr, lr_min_factor=1e-3):
        self.steps = steps
        self.base_lr = base_lr
        self.lr_min_factor = lr_min_factor

    def __call__(self, step):
        return self.base_lr * (self.lr_min_factor +
                               (1 - self.lr_min_factor) * 0.5 *
                               (1 + np.cos((step / self.steps) * np.pi)))

def loadEncoder(model, path):
    state_dict = torch.load(path)['state_dict']
    for k,v in model.state_dict().items():
        if 'module.'+k in state_dict:
           model.state_dict()[k].copy_(state_dict['module.'+k])
        else:
           print(k)
    return model

def train(train_loader, val_loader, target_model, lowRankLL, classifier, optimizer, scheduler, criterion):
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

    target_model.eval()
    lowRankLL.eval()
    for epoch in range(intial_epoch, nb_epoch):
        scheduler.step(epoch)

        classifier.train()
        for batch_ndx, (x_pre, y) in enumerate(train_loader):
            x_pre, y = x_pre.float().to(device), y.float().to(device)
            
            with torch.no_grad():
               _, features = target_model(x_pre, -1)
               outputFeatures = lowRankLL(features)
           
            pre_out = classifier(outputFeatures)[:,0]

            loss = criterion(pre_out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))
            torch.save({
              'epoch': epoch+1,
              'state_dict' : classifier.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()
            },os.path.join(conf.model_path, str(epoch)+".pt"))

            if (batch_ndx + 1) % 3 ==0:
               print('Epoch [{}/{}], batch_ndx {}, Loss: {:.6f}'.format(epoch + 1, conf.nb_epoch, batch_ndx + 1, np.average(train_losses)))
               sys.stdout.flush()

        correct = 0
        total = 0
        classifier.eval()
        with torch.no_grad():
             for batch_ndx, (x_pre, y) in enumerate(val_loader):
                 x_pre, y = x_pre.float().to(device), y.float().to(device)
               
                 _, features = target_model(x_pre, -1)
                 outputFeatures = lowRankLL(features)
            
                 pre_out = classifier(outputFeatures)[:,0]

                 loss = criterion(pre_out, y)
                 valid_losses.append(loss.item())

                 predicted = torch.round(pre_out)

                 total += y.size(0)
                 correct += (predicted == y.to(device)).sum()
             print('Accuracy: %.2f %%' % (100 * float(correct) / total))


def valiate(target_model, val_loader):
    correct = 0
    total = 0
    target_model.eval()
    with torch.no_grad():
        print("validating....")
        for batch_ndx, (x_pre, y) in enumerate(val_loader):
            #x_pre0, x_pre1, y = x_pre[0].float().to(device), x_pre[1].float().to(device), y.float().to(device)
            #pre_out = target_model([x_pre0, x_pre1], t)[:,0]

            x_pre, y = x_pre.float().to(device), y.float().to(device)
            pre_out = target_model(x_pre, -1)[:,0]
            loss = criterion(pre_out, y)

            predicted = torch.round(pre_out)

            total += y.size(0)
            correct += (predicted == y.to(device)).sum()
        print('Accuracy: %.2f %%' % (100 * float(correct) / total))

if __name__ == "__main__":
  dataset_train = KrasClassificationDataset('/scratch/unath/KRAS',  trainDataset=True, train=False)
  train_loader = DataLoader(dataset_train, batch_size=8, shuffle=False, num_workers=8)

  dataset_val = KrasClassificationDataset('/scratch/unath/KRAS', trainDataset=False,  train=False)
  val_loader = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=8)


  base_model = unet3d.UNet3DClassification()

  base_model = loadEncoder(base_model, 'pretrained_weights/Y90Genesis_Chest_CT.pt')
  target_model = TargetNetV2(base_model, n_class=1)
  model_path = "/data/yyang409/unath/pretrained_weights/classification/kras-unet-standard-v3/4.pt"
  target_model = loadEncoder(target_model, model_path)
  rank = 42

  lowrankLL = LowRankLinear(1024, rank)
  lowrankLL.load_state_dict(torch.load("/data/yyang409/unath/pretrained_weights/classification/lowRankLL/best_checkpoint_42.pt"))
  lowrankLL.eval()
  classifier = Classifier(rank)


  device = 'cuda'
  target_model.to(device)
  target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])
  lowrankLL.to(device)
  lowrankLL = nn.DataParallel(lowrankLL, device_ids = [i for i in range(torch.cuda.device_count())])
  classifier.to(device)
  classifier = nn.DataParallel(classifier, device_ids = [i for i in range(torch.cuda.device_count())])
  target_model.load_state_dict(torch.load(model_path)['state_dict'])
  target_model.eval()
  lowrankLL.eval()
  
  # /data/yyang409/unath/pretrained_weights/classification/lowRankLL/best_checkpoint_42.pt
  criterion = nn.BCELoss()
  optimizer = torch.optim.SGD(classifier.parameters(), 0.1, momentum=0.9, weight_decay=0.0, nesterov=False)
  #optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)

  scheduler_func = CosineScheduler(100, 0.1, 1e-4)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_func)

  train(train_loader, val_loader, target_model, lowrankLL, classifier,  optimizer, scheduler, criterion)
  #valiate(target_model, val_loader)

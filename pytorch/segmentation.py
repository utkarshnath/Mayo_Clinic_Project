import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d
import csv
import numpy as np
import nibabel as nib
import cv2

from torchsummary import summary
import sys
from utils import *
import unet3d
from config import models_genesis_config
from tqdm import tqdm
import torchvision.transforms as transforms

conf = models_genesis_config()
conf.display()

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2 # 400 to -300
    X = torch.clamp(img, min=lower, max=upper)
    X = X - X.min()
    X = X / X.max()
    #X = (X*255.0).astype('uint8')
    return X

class Y90Dataset(Dataset):

    def __init__(self, pathImageDirectory, csvFilePath, train=True, dim=(224, 224, 3), mean=0.4142, std=0.3993):
        self.pathImageDirectory = pathImageDirectory
        self.csvFilePath = csvFilePath
        # self.transforms = transforms
        self.dim = dim
        self.mean = mean
        self.std = std
        self.img = []
        self.mask = []

        with open(csvFilePath, 'r') as read_obj:
             csv_reader = csv.reader(read_obj)
             self.item_list = list(csv_reader)
        channel = 0
        for i in range(len(self.item_list)):
            img_path = self.item_list[i][1]
            mask_path = self.item_list[i][2]
            img = nib.load(pathImageDirectory + '/' + img_path).get_fdata()
            mask = nib.load(pathImageDirectory + '/' + mask_path).get_fdata()
            self.img.append(img)
            self.mask.append(mask)
            #if img.shape[2]>channel:
            channel += img.shape[2]
            index = (torch.tensor(mask).sum((0,1)) > 0).nonzero()
            #print(index[0], index[-1], img.shape[2])
        #print("channel", channel/len(self.item_list))

        assert len(self.img)==len(self.mask)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        mask = self.mask[idx]

        #img = window(img)
        #img = img/255.0

        #padding to make 144
        
        img = cv2.resize(img, (160, 160))
        mask = cv2.resize(mask, (160, 160))
        
        img = torch.tensor(img)
        mask = torch.tensor(mask)
        pad_length = 112 - img.shape[2]
        pad = (0, pad_length)
        if pad_length > 0:
           img = F.pad(img, pad, "constant", 0)
           mask = F.pad(mask, pad, "constant", 0)
     
           x = img[None,:]
           y = mask[None,:]
        else:
           x = img[None,:,:,:112]
           y = mask[None,:,:,:112]
        
        x = window(x)
        #x = torch.clamp(x, min=-0.0, max=4000.0)
        #x = x/4000.0
        
        return x, y

class CosineScheduler:
    def __init__(self, steps, base_lr, lr_min_factor=1e-3):
        self.steps = steps
        self.base_lr = base_lr
        self.lr_min_factor = lr_min_factor

    def __call__(self, step):
        return self.base_lr * (self.lr_min_factor +
                               (1 - self.lr_min_factor) * 0.5 *
                               (1 + np.cos((step / self.steps) * np.pi)))


#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

# prepare your own data
dataset_train = Y90Dataset('/scratch/unath/Y90-Project/train/train_data', '/scratch/unath/Y90-Project/train/train_data.csv')
dataset_val = Y90Dataset('/scratch/unath/Y90-Project/test/test_data', '/scratch/unath/Y90-Project/test/test_data.csv')

train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=4)

# prepare the 3D model

model = unet3d.UNet3D()

#Load pre-trained weights
weight_dir = '/home/unath/medical_imaging_projects/Models_Genesis/Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
model.load_state_dict(unParalled_state_dict)

device = 'cuda'
model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = torch_dice_coef_loss
optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)
scheduler_func = CosineScheduler(100, 0.1, 1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_func)

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
best_loss = 100000
intial_epoch =0
num_epoch_no_improvement = 0
sys.stdout.flush()


for epoch in range(0, conf.nb_epoch):
    scheduler.step(epoch)
    model.train()
    for batch_ndx, (x,y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(round(loss.item(), 2))
        if (batch_ndx + 1) % 5 ==0:
            print('Epoch [{}/{}], batch_ndx {}, Loss: {:.6f}'
                .format(epoch + 1, conf.nb_epoch, batch_ndx + 1, np.average(train_losses)))
            sys.stdout.flush()

    with torch.no_grad():
        model.eval()
        print("validating....")
        for batch_ndx, (x,y) in enumerate(val_loader):
            x, y = x.float().to(device), y.float().to(device)
            pred = model(x)
            loss = criterion(pred, y)
            valid_losses.append(loss.item())

    #logging
    train_loss=np.average(train_losses)
    valid_loss=np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
    train_losses=[]
    valid_losses=[]
    if valid_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0
        #save model
        torch.save({
            'epoch': epoch+1,
            'state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },os.path.join(conf.model_path, "Y90Genesis_Chest_CT-run3.pt"))
        print("Saving model ",os.path.join(conf.model_path,"Y90Genesis_Chest_CT-run3.pt"))
    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    if num_epoch_no_improvement == conf.patience:
        print("Early Stopping")
        break
    sys.stdout.flush()

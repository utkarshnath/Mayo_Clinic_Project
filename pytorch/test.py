from torchsummary import summary
import sys
from utils import *
import unet3d
from config import models_genesis_config
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


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

conf = models_genesis_config()
conf.display()

#from segmentation import Y90Dataset

class Y90Dataset(Dataset):

    def __init__(self, pathImageDirectory, csvFilePath, dim=(224, 224, 3), mean=0.4142, std=0.3993):
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
        self.max_channel = 112
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
        pad_length = self.max_channel - img.shape[2]
        pad = (0, pad_length)
        if pad_length > 0:
           img = F.pad(img, pad, "constant", 0)
           mask = F.pad(mask, pad, "constant", 0)

           x = img[None,:]
           y = mask[None,:]
        else:
           x = img[None,:,:,:self.max_channel]
           y = mask[None,:,:,:self.max_channel]

        return x, y

def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def mean_dice_coef(y_true,y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    sum = 0
    c = 0
    for i in range (y_true.shape[0]):
        p = 0
        curr_sum = 0
        for j in range(112):
            if y_true[i,0,:,:,j].sum()>0:
               p += 1
               curr_sum += dice(y_true[i,0,:,:,j],y_pred[i,0,:,:,j])
            if p>0:
               curr_sum = curr_sum/p
               sum += curr_sum
        if y_true[i,0,:,:,:].sum()==0:
           c+=1
    return sum/(y_true.shape[0]-c)

def mean_dice_coef_all(y_true,y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    #y_pred = y_pred > 0.5
    sum = 0
    for i in range (y_true.shape[0]):
        curr_sum = 0
        for j in range(112):
               curr_sum += dice(y_true[i,0,:,:,j],y_pred[i,0,:,:,j])
        curr_sum = curr_sum/112
        sum += curr_sum
    return sum/(y_true.shape[0])

#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

#dataset_train = Y90Dataset('/scratch/unath/Y90-Project/train/train_data', '/scratch/unath/Y90-Project/train/train_data.csv')
#train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True)

dataset_val = Y90Dataset('/scratch/unath/Y90-Project/test/test_data', '/scratch/unath/Y90-Project/test/test_data.csv')
val_loader = DataLoader(dataset_val, batch_size=4)

model = unet3d.UNet3D()

#Load pre-trained weights
weight_dir = '/home/unath/medical_imaging_projects/ModelsGenesis/pytorch/pretrained_weights/Y90Genesis_Chest_CT.pt'
weight_dir = '/scratch/unath/pretrained_weights/rotation_window_cosine_run2/Y90Genesis_Chest_CT-run3.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
model.load_state_dict(unParalled_state_dict)

device = 'cuda'
model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=0.0, nesterov=False)
criterion = torch_dice_coef_loss

# train the model
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)
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

score = 0
model.eval()
for batch_ndx, (x,y) in enumerate(val_loader):
    x, y = x.float().to(device), y.float().to(device)
    with torch.no_grad(): 
       pred = model(x)
    print(pred.shape, y.shape)
    torch.save(pred, 'results/pred/'+str(batch_ndx)+'.pt')
    torch.save(y, 'results/target/'+str(batch_ndx)+'.pt')
    #torch.save(pred, "results/"+str(batch_ndx)+" pred.pt")
    #torch.save(y, "results/"+str(batch_ndx)+" y.pt")
    #print("shape", pred.shape[4])
    
    #for i in range(pred.shape[4]):
    #    if pred[1,0,:,:,i].sum()>0:
    #       print(pred[1,0,:,:,i].sum())
    
    score +=  mean_dice_coef(y, pred)

print("score ", score)
print("batch_ndx", batch_ndx)


'''
#Load pre-trained weights
weight_dir = '/home/unath/medical_imaging_projects/ModelsGenesis/pytorch/pretrained_weights/Y90Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
model.load_state_dict(unParalled_state_dict)


device = 'cuda'
model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])

model.eval()

score = 0
for batch_ndx, (x,y) in enumerate(val_loader):
    x, y = x.float().to(device), y.float().to(device)
    pred = model(x)
    print(batch_ndx)
    #score +=  mean_dice_coef(y, pred)

print("score ", score)
print("batch_ndx", batch_ndx)
'''

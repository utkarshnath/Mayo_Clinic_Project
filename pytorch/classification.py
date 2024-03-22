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
from dataset import Y90Dataset,Y90WeightedDataset 
from models import TargetNet, MultiHeadNet, JointHeadNet, WeightedUnet3D   


parser = argparse.ArgumentParser(description='Y90 - Project')
parser.add_argument('--batch_size', type=int, default=12, help='')
parser.add_argument('--model_type', type=int, default=0, help='0 for standard, 1 for multiple-head, 2 for joint model')
parser.add_argument('--train_type', type=int, default=0, help='0 for training, 1 for testing, 2 for series prediction')
parser.add_argument('--checkpoint', type=str, default='/data/yyang409/unath/pretrained_weights/classification/jointtraining/Y90Genesis_classification.pt', help='checkpoint path for testing/inference')

args = parser.parse_args()


conf = models_genesis_config()
conf.display()

       
def get_patient_dict():
    patient_key_path = "/scratch/unath/Y90-Project/patientKey/key_table.csv"
    patient_dict = {}
    with open(patient_key_path, 'r') as read_obj:
       csv_reader = csv.reader(read_obj)
       item_list = list(csv_reader)
    p_count = 0
    for i in range(len(item_list)):
        if i==0:
           continue
        key = item_list[i][2]
        if item_list[i][4]=='100':
           value = 0
        else:
           value = 1
           p_count+=1
        patient_dict[key] = value
    print(p_count, len(item_list))
    return patient_dict


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

def train(train_loader, val_loader, target_model, args):
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3,0.7]).to(device))
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(target_model.parameters(), 0.1, momentum=0.9, weight_decay=0.0, nesterov=False)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    scheduler_func = CosineScheduler(100, 0.1, 1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_func)

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
        if args.model_type==2:
           for batch_ndx, (x,y, series_label) in enumerate(train_loader):
               x, y = x.float().to(device), y.float().to(device)
               pred = torch.sigmoid(target_model(x, series_label, temperature))
            
               loss = criterion(pred[:,0], y)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               train_losses.append(round(loss.item(), 2))
               if (batch_ndx + 1) % 5 ==0:
                  print('Epoch [{}/{}], batch_ndx {}, Loss: {:.6f}'
                   .  format(epoch + 1, conf.nb_epoch, batch_ndx + 1, np.average(train_losses)))
                  sys.stdout.flush()

           correct = 0
           total = 0
           with torch.no_grad():
              target_model.eval()
              print("validating....")
              for batch_ndx, (x, y, series_label) in enumerate(val_loader):
                  x, y = x.float().to(device), y.float().to(device)
                  pred = torch.sigmoid(target_model(x, series_label, temperature))
                  loss = criterion(pred[:,0], y)
                  valid_losses.append(loss.item())
                  predicted = torch.round(pred[:,0])

                  total += y.size(0)
                  correct += (predicted == y.to(device)).sum()
              print('Accuracy: %.2f %%' % (100 * float(correct) / total))

        else:
           for batch_ndx, (x,y) in enumerate(train_loader):
               x, y = x.float().to(device), y.float().to(device)
               if args.model_type==1:
                  pred = torch.sigmoid(target_model(x, temperature))
               else:
                  pred = torch.sigmoid(target_model(x))

               loss = criterion(pred[:,0], y)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               train_losses.append(round(loss.item(), 2))
               if (batch_ndx + 1) % 5 ==0:
                  print('Epoch [{}/{}], batch_ndx {}, Loss: {:.6f}'
                   .  format(epoch + 1, conf.nb_epoch, batch_ndx + 1, np.average(train_losses)))
                  sys.stdout.flush()  

           correct = 0
           total = 0
           with torch.no_grad():
              target_model.eval()
              print("validating....")
              for batch_ndx, (x, y) in enumerate(val_loader):
                  x, y = x.float().to(device), y.float().to(device)
                  if args.model_type==1:
                     pred = torch.sigmoid(target_model(x, temperature))
                  else:
                     pred = torch.sigmoid(target_model(x))
                  loss = criterion(pred[:,0], y)
                  valid_losses.append(loss.item())
                  predicted = torch.round(pred[:,0])

                  total += y.size(0)
                  correct += (predicted == y.to(device)).sum()
              print('Accuracy: %.2f %%' % (100 * float(correct) / total))

        temperature = temperature * np.exp(-0.045)

        #logging
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
        train_losses=[]
        valid_losses=[]
        torch.save({
              'epoch': epoch+1,
              'state_dict' : target_model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()
           },os.path.join(conf.model_path, "Y90Genesis_classification"+str(epoch)+".pt"))
        if valid_loss < best_loss:
           print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
           best_loss = valid_loss
           num_epoch_no_improvement = 0
           #save model
           torch.save({
              'epoch': epoch+1,
              'state_dict' : target_model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()
           },os.path.join(conf.model_path, "Y90Genesis_classification.pt"))
           print("Saving model ",os.path.join(conf.model_path,"Y90Genesis_classification.pt"))
        else:
           print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
           num_epoch_no_improvement += 1
        if num_epoch_no_improvement == conf.patience:
           print("Early Stopping")
           break
        sys.stdout.flush()


def test(model, path, val_loader):
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    model.load_state_dict(unParalled_state_dict)

    device = 'cuda'
    model.to(device)
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])    

    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        print("validating....")
        for batch_ndx, (x,y) in enumerate(val_loader):
            x, y = x.float().to(device), y.to(device)
            pred = torch.sigmoid(model(x))
            predicted = torch.round(pred[:,0])
            #_, predicted = torch.max(pred.data, 1)
            print(predicted, y)
            total += y.size(0)
            correct += (predicted == y.to(device)).sum()
        print('Accuracy: %.2f %%' % (100 * float(correct) / total))
    
def get_series_prediction(model, path):
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    model.load_state_dict(unParalled_state_dict)

    device = 'cuda'
    model.to(device)
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])

    series = {'ph0', 'ph1', 'ph2', 'ph3', 'adc', 't2'}
    prediction = {}
    gt = {}
    for key in series:
        dataset_val = Y90Dataset('/scratch/unath/Y90-Project/test/test_data', '/scratch/unath/Y90-Project/test/test_' + key +'.csv', patient_dict, train_type=2, series_wise_pred=True)
        val_loader = DataLoader(dataset_val, batch_size=12)
        with torch.no_grad():
             model.eval() 
             for batch_ndx, (x,y, p_id) in enumerate(val_loader):
                  x, y = x.float().to(device), y.to(device)
                  pred = torch.sigmoid(model(x))
                  predicted = torch.round(pred[:,0])
                  for i in range(x.shape[0]):
                      if p_id[i] in prediction:
                         prediction[p_id[i]].append(predicted[i].data)
                      else:
                         prediction[p_id[i]] = [predicted[i].data]

                      if p_id[i] in gt:
                         gt[p_id[i]].append(y[i].data)
                      else:
                         gt[p_id[i]] = [y[i].data]

    for key in prediction.keys():
        print(key," : ", prediction[key])
        print(key," : ", gt[key])
        print()




if __name__ == "__main__":
   patient_dict = get_patient_dict()
   base_model = unet3d.UNet3DClassification()
   base_model = loadEncoder(base_model, 'pretrained_weights/Y90Genesis_Chest_CT.pt')

   if args.model_type == 0:
      target_model = TargetNet(base_model, n_class=1)
   elif args.model_type == 1:
      target_model = MultiHeadNet(base_model, n_class=1)
   elif args.model_type == 2:
      target_model = JointHeadNet(base_model, n_class=1)
   elif args.model_type == 3:
      target_model = WeightedUnet3D(base_model, n_class=1)


   device = 'cuda'
   target_model.to(device)
   target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])

   if args.train_type==0:
      #series = 'ph0'
      series = 'data'
      if args.model_type != 3:
         dataset_train = Y90Dataset('/scratch/unath/Y90-Project/train/train_data', '/scratch/unath/Y90-Project/train/train_' + series +'.csv', patient_dict, train_type=0)
         dataset_val = Y90Dataset('/scratch/unath/Y90-Project/test/test_data', '/scratch/unath/Y90-Project/test/test_' + series +'.csv', patient_dict, train_type=1)
      else:
        dataset_train = Y90WeightedDataset('/scratch/unath/Y90-Project/train/train_data', '/scratch/unath/Y90-Project/train/train_' + series +'.csv', patient_dict, train_type=0)
        dataset_val = Y90WeightedDataset('/scratch/unath/Y90-Project/test/test_data', '/scratch/unath/Y90-Project/test/test_' + series +'.csv', patient_dict, train_type=1) 

      train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
      val_loader = DataLoader(dataset_val, batch_size=args.batch_size)
      train(train_loader, val_loader, target_model, args)
   elif args.train_type==1:
      series = 'data'
      if args.model_type != 3:
         dataset_val = Y90Dataset('/scratch/unath/Y90-Project/test/test_data', '/scratch/unath/Y90-Project/test/test_' + series +'.csv', patient_dict, train_type=1)
      else:
         dataset_val = Y90WeightedDataset('/scratch/unath/Y90-Project/test/test_data', '/scratch/unath/Y90-Project/test/test_' + series +'.csv', patient_dict, train_type=1)

      val_loader = DataLoader(dataset_val, batch_size=args.batch_size)
      test(target_model, args.checkpoint, val_loader)
   elif args.train_type==2:
      get_series_prediction(model, args.checkpoint) 


'''

   patient_dict = get_patient_dict()
   # prepare your own data
   #series = 'ph0'
   series = 'data'
   dataset_train = Y90Dataset('/scratch/unath/Y90-Project/train/train_data', '/scratch/unath/Y90-Project/train/train_' + series +'.csv', patient_dict, train_type=0)
   dataset_val = Y90Dataset('/scratch/unath/Y90-Project/test/test_data', '/scratch/unath/Y90-Project/test/test_' + series +'.csv', patient_dict, train_type=1)

   train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
   val_loader = DataLoader(dataset_val, batch_size=args.batch_size)

   model = unet3d.UNet3DClassification()
   model = TargetNet(model, n_class=1)
   #model = MultiHeadNet(model, n_class=1)
   #model = JointHeadNet(model, n_class=1)
   #path = "pretrained_weights/Y90Genesis_classification.pt"
   path = "/data/yyang409/unath/pretrained_weights/rotation_window_cosine/Y90Genesis_classification.pt" 

   #path = "/scratch/unath/pretrained_weights/classification/jointtraining/Y90Genesis_classification.pt" 
   #get_series_prediction(model, path)
   #test(model, path, val_loader) 
   train(train_loader, val_loader, args)
'''

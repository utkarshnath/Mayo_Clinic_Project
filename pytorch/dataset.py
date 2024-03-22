import cv2
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
from PIL import Image
import nibabel as nib
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
#from preprocessing import get_patiest_ids 
import csv
import os
import shutil
import SimpleITK as sitk

#def window(img, WL=50, WW=350):
#    upper, lower = WL+WW//2, WL-WW//2 # 400 to -300
#    X = torch.clamp(img, min=lower, max=upper)
#    X = X - X.min()
#    X = X / X.max()
#    #X = (X*255.0).astype('uint8')
#    return X

def get_left_right_idx_should_pad(target_size, dim, test=False):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        if test:
            left = pad_extent//2
        else:
            left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim, test=False):
    if dim > target_size:
        if test:
            crop_extent = dim - target_size
            left = crop_extent//2
            right = crop_extent - left
        else:
            crop_extent = dim - target_size
            left = random.randint(0, crop_extent)
            right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)

def pad_or_crop_image(image, seg=None, target_size=(128, 144, 144), test=False):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim, test=test) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim, test=test) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image

def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2 # 400 to -300
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

def mri_series_label(mask_path):
    if mask_path.find('ph0')!=-1:
       return 0
    if mask_path.find('ph1')!=-1:
       return 1
    if mask_path.find('ph2')!=-1:
       return 2
    if mask_path.find('ph3')!=-1:
       return 3
    if mask_path.find('adc')!=-1:
       return 4
    if mask_path.find('t2')!=-1:
       return 5
    return 6


class Y90Dataset(Dataset):

    def __init__(self, pathImageDirectory, csvFilePath, patient_dict, train_type=0, model_type=False, dim=(224, 224, 3), mean=0.4142, std=0.3993):
        self.pathImageDirectory = pathImageDirectory
        self.csvFilePath = csvFilePath
        self.dim = dim
        self.mean = mean
        self.std = std
        self.img = []
        self.label = []
        self.series_label = []
        self.train_type = train_type
        self.model_type = model_type
        self.patient_id = []

        with open(csvFilePath, 'r') as read_obj:
             csv_reader = csv.reader(read_obj)
             self.item_list = list(csv_reader)
        channel = 0
        for i in range(len(self.item_list)):
            img_path = self.item_list[i][1]
            mask_path = self.item_list[i][2]
            self.series_label.append(mri_series_label(mask_path))
            patient_id = img_path.split('_')[0]
            if patient_id.find('14c74e06')!=-1:
               patient_id = '14c74e06'
            label = patient_dict[patient_id]
            self.label.append(label)
            img = nib.load(pathImageDirectory + '/' + img_path).get_fdata()
            self.img.append(img)
            self.patient_id.append(patient_id)

        transformations_list = []
        #transformations_list.append(transforms.Resize((160, 160)))
        if self.train_type==0:
           transformations_list.append(transforms.RandomRotation(degrees=(-30, 30)))
        #transformations_list.append(transforms.ToTensor())
        self.transformSequence = transforms.Compose(transformations_list)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]

        img = cv2.resize(img, (160, 160))

        #img = Image.fromarray(img)
        img = self.transformSequence(torch.tensor(img))

        #img = torch.tensor(img)
        pad_length = 112 - img.shape[2]
        pad = (0, pad_length)
        if pad_length > 0:
           img = F.pad(img, pad, "constant", 0)

           x = img[None,:]
        else:
           x = img[None,:,:,:112]
        x = window(x)
        #x = torch.clamp(x, min=-0.0, max=4000.0)
        #x = x/4000.0
        #print(x.min(), x.max())

        if self.model_type==2 and self.train_type==2:
           return x, torch.tensor(label, dtype=torch.long), self.series_label[idx], self.patient_id[idx]
        elif self.train_type==2:
           return x, torch.tensor(label, dtype=torch.long), self.patient_id[idx]
        elif self.model_type==2:
           return x, torch.tensor(label, dtype=torch.long), self.series_label[idx]
        else:
           return x, torch.tensor(label, dtype=torch.long)

'''
class Y90WeightedDataset(Dataset):

    def __init__(self, pathImageDirectory, csvFilePath, patient_dict, train_type=0, model_type=False, dim=(224, 224, 3), mean=0.4142, std=0.3993):
        self.pathImageDirectory = pathImageDirectory
        self.csvFilePath = csvFilePath
        self.patient_mri = {}  # key(patient_id), value [all mri scans]
        self.dim = dim
        self.mean = mean
        self.std = std
        self.img = []
        self.label = []
        self.series_label = []
        self.train_type = train_type
        self.model_type = model_type

        self.patient_id = get_patiest_ids(csvFilePath)
        if train_type==1:
           self.patient_id.remove('14c74e06-Y2')
           self.patient_id.remove('14c74e06')

        with open(csvFilePath, 'r') as read_obj:
             csv_reader = csv.reader(read_obj)
             self.item_list = list(csv_reader)

        for p_id in self.patient_id:
            if p_id.find('14c74e06')!=-1:
               continue
            for items in self.item_list:
                img_path = items[1]
                if img_path.find(p_id)!=-1:
                    img = nib.load(pathImageDirectory + '/' + img_path).get_fdata()
                    if p_id in self.patient_mri:
                       self.patient_mri[p_id].append(img)
                    else:
                       self.patient_mri[p_id] = [img]

            label = patient_dict[p_id]
            self.label.append(label)

        transformations_list = []
        #if self.train:
        #   transformations_list.append(transforms.RandomRotation(degrees=(-30, 30)))
        self.transformSequence = transforms.Compose(transformations_list)

    def __len__(self):
        return len(self.label)

     
    def __getitem__(self, idx):
        p_id = self.patient_id[idx]
        images = None
        label = self.label[idx]

        for i in range(len(self.patient_mri[p_id])):
            img = self.patient_mri[p_id][i]
            img = cv2.resize(img, (160, 160))
            img = self.transformSequence(torch.tensor(img))
        
            pad_length = 112 - img.shape[2]
            pad = (0, pad_length)
            if pad_length > 0:
               img = F.pad(img, pad, "constant", 0)

               x = img[None,:]
            else:
               x = img[None,:,:,:112]
            x = window(x)
            if images==None:
               images = x
            else:
               images = torch.cat((images, x))
        
        n_series = images.shape[0]
        if n_series<7:
           pad = torch.zeros(7-n_series, 160, 160, 112)
           images = torch.cat((images, pad)) 
        return images[:7], label

'''
class Y90PrePostSegDataset(Dataset):

    def __init__(self, pathImageDirectory, csvFilePath, patient_dict, train_type=0, model_type=False, dim=(224, 224, 3), mean=0.4142, std=0.3993):
        self.pathImageDirectory = pathImageDirectory
        self.seg_dir = "/data/yyang409/unath/Y90-Project/segmentation_predictions/"
        self.csvFilePath = csvFilePath
        key_path = '/data/yyang409/unath/Y90-Project/patientKey/key_table.csv' 
        self.patient_mri_pre = {}
        self.patient_mri_post = {}

        self.dim = dim
        self.mean = mean
        self.std = std
        self.img = []
        self.label = []
        self.series_label = []
        self.train_type = train_type
        self.model_type = model_type

        absent_train_ids = {'d5787b11','c4581bcd','a6ed6ca0','42d6fea0','6dc3617d','93d9fcd1','ba243a54','28694675',
                           'bf4f739a','7357d1e5','193cbaa2','9b8d657d','92c5db51','a652706d','31c49e50','0fa97555',
                            'aaab8b06-Y1','aaab8b06-Y2','e37c074e-Y1','e37c074e','2','1276f945','a4c0ec2f','8726c2b3','18bc66b6','f10ba286-Y2'}

        absent_test_ids = {'2','1276f945','a4c0ec2f','8726c2b3','18bc66b6'}

        self.pre_pid = []
        self.post_pid = []
        self.label = []
        with open(key_path, 'r') as read_obj:
             csv_reader = csv.reader(read_obj)
             self.item_list = list(csv_reader)

        # [53,end] test
        # [1,14) test
        for i in range(1, len(self.item_list), 2):
            if train_type==0 and int(self.item_list[i][0])<14:
                  continue
            elif train_type==1 and int(self.item_list[i][0])>=14:
               break
            pre = self.item_list[i][2]
            post = self.item_list[i+1][2]
            
            if (pre in absent_train_ids) or (pre in absent_test_ids) or (post in absent_train_ids) or (post in absent_test_ids):
                continue

            self.pre_pid.append(pre)
            self.post_pid.append(post)

            print(self.item_list[i][4])
 
            if int(self.item_list[i][4])==100:
               self.label.append(1)
            elif int(self.item_list[i][4])>=90 and int(self.item_list[i][4])<100:
               self.label.append(0)
            else:
               self.label.append(0)

        print(self.pre_pid)
        print(self.post_pid)
        print(self.label)



        with open(csvFilePath, 'r') as read_obj:
             csv_reader = csv.reader(read_obj)
             self.item_list = list(csv_reader)

        self.pre_imgage_path = []
        self.post_image_path = []

        for i in range(len(self.pre_pid)):
            #if i>10:
            #   break
            for items in self.item_list:
                img_path = items[1]
                if img_path.find(self.pre_pid[i])!=-1:
                    #if not os.path.exists('test_data/pre/'+self.pre_pid[i]):
                    #   os.mkdir('test_data/pre/'+self.pre_pid[i])
                    #shutil.copyfile(pathImageDirectory + '/' + items[2], '/home/unath/medical_imaging_projects/ModelsGenesis/pytorch/test_data/pre/'+self.pre_pid[i]+'/'+items[2])
                    img = self.load_nii(pathImageDirectory + '/' + img_path)
                    
                    if self.pre_pid[i] in self.patient_mri_pre:
                       self.patient_mri_pre[self.pre_pid[i]].append([img,img_path])
                    else:
                       self.patient_mri_pre[self.pre_pid[i]] = [[img,img_path]]

            for items in self.item_list:
                img_path = items[1]
                if img_path.find(self.post_pid[i])!=-1:
                    #if not os.path.exists('test_data/post/'+self.post_pid[i]):
                    #   os.mkdir('test_data/post/'+self.post_pid[i])
                    #shutil.copyfile(pathImageDirectory + '/' + items[2], '/home/unath/medical_imaging_projects/ModelsGenesis/pytorch/test_data/post/'+self.post_pid[i]+'/'+items[2])
                    img = self.load_nii(pathImageDirectory + '/' + img_path)
                    
                    if self.post_pid[i] in self.patient_mri_post:
                       self.patient_mri_post[self.post_pid[i]].append([img,img_path])
                    else:
                       self.patient_mri_post[self.post_pid[i]] = [[img,img_path]]


        transformations_list = []
        #if self.train:
        #   transformations_list.append(transforms.RandomRotation(degrees=(-30, 30)))
        self.transformSequence = transforms.Compose(transformations_list)

    def __len__(self):
        #return 10
        return len(self.label)

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __getitem__(self, idx):
        pre_pid = self.pre_pid[idx]
        post_pid = self.post_pid[idx]
        pre_images = None
        post_images = None
        label = self.label[idx]

        dim = 128
        for i in range(len(self.patient_mri_pre[pre_pid])):
            img = self.patient_mri_pre[pre_pid][i][0]
            mri_scan = window(img)

            patient_image = mri_scan[None,:]

            
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)

            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128
            patient_image = pad_or_crop_image(patient_image, None, target_size=(dim, dim, dim))

            patient_image = patient_image.astype("float16")
            patient_image = torch.from_numpy(patient_image)

            seg_pred_path = self.seg_dir + self.patient_mri_pre[pre_pid][i][1].split('.')[0]+'.pt'
            seg_pred = torch.load(seg_pred_path, map_location='cpu')
            if pre_images==None:
               pre_images = patient_image * seg_pred[None,:]
            else:
               pre_images = torch.cat((pre_images, patient_image * seg_pred[None,:]))


        for i in range(len(self.patient_mri_post[post_pid])):
            img = self.patient_mri_post[post_pid][i][0]
            mri_scan = window(img)

            patient_image = mri_scan[None,:]


            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)

            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128
            patient_image = pad_or_crop_image(patient_image, None, target_size=(dim, dim, dim))

            patient_image = patient_image.astype("float16")
            patient_image = torch.from_numpy(patient_image)

            seg_pred_path = self.seg_dir + self.patient_mri_post[post_pid][i][1].split('.')[0]+'.pt'
            seg_pred = torch.load(seg_pred_path, map_location='cpu')
                 
            if post_images==None:
               post_images = patient_image * seg_pred[None,:]
            else:
               post_images = torch.cat((post_images, patient_image * seg_pred[None,:]))


        n_series = pre_images.shape[0]
        if n_series<7:
           pad = torch.zeros(7-n_series, dim, dim, dim)
           pre_images = torch.cat((pre_images, pad))

        n_series = post_images.shape[0]
        if n_series<7:
           pad = torch.zeros(7-n_series, dim, dim, dim)
           post_images = torch.cat((post_images, pad))

        return pre_images[:7], post_images[:7], label



class Y90PrePostDataset(Dataset):

    def __init__(self, pathImageDirectory, csvFilePath, patient_dict, train_type=0, model_type=False, dim=(224, 224, 3), mean=0.4142, std=0.3993):
        self.pathImageDirectory = pathImageDirectory
        self.csvFilePath = csvFilePath
        key_path = '/data/yyang409/unath/Y90-Project/patientKey/key_table.csv'
        self.patient_mri_pre = {}
        self.patient_mri_post = {}

        self.dim = dim
        self.mean = mean
        self.std = std
        self.img = []
        self.label = []
        self.series_label = []
        self.train_type = train_type
        self.model_type = model_type

        absent_train_ids = {'d5787b11','c4581bcd','a6ed6ca0','42d6fea0','6dc3617d','93d9fcd1','ba243a54','28694675',
                           'bf4f739a','7357d1e5','193cbaa2','9b8d657d','92c5db51','a652706d','31c49e50','0fa97555',
                            'aaab8b06-Y1','aaab8b06-Y2','e37c074e-Y1','e37c074e','2','1276f945','a4c0ec2f','8726c2b3','18bc66b6','f10ba286-Y2'}

        absent_test_ids = {'2','1276f945','a4c0ec2f','8726c2b3','18bc66b6'}

        self.pre_pid = []
        self.post_pid = []
        self.label = []
        with open(key_path, 'r') as read_obj:
             csv_reader = csv.reader(read_obj)
             self.item_list = list(csv_reader)

        # [53,end] test
        # [1,14) test
        for i in range(1, len(self.item_list), 2):
            if train_type==0 and int(self.item_list[i][0])<14:
                  continue
            elif train_type==1 and int(self.item_list[i][0])>=14:
               break
            pre = self.item_list[i][2]
            post = self.item_list[i+1][2]

            if (pre in absent_train_ids) or (pre in absent_test_ids) or (post in absent_train_ids) or (post in absent_test_ids):
                continue

            self.pre_pid.append(pre)
            self.post_pid.append(post)

            print(self.item_list[i][4])

            if int(self.item_list[i][4])==100:
               self.label.append(2)
            elif int(self.item_list[i][4])>=90 and int(self.item_list[i][4])<100:
               self.label.append(1)
            else:
               self.label.append(0)

        print(self.pre_pid)
        print(self.post_pid)



        with open(csvFilePath, 'r') as read_obj:
             csv_reader = csv.reader(read_obj)
             self.item_list = list(csv_reader)

        for i in range(len(self.pre_pid)):
            for items in self.item_list:
                img_path = items[1]
                if img_path.find(self.pre_pid[i])!=-1:
                    #if not os.path.exists('test_data/pre/'+self.pre_pid[i]):
                    #   os.mkdir('test_data/pre/'+self.pre_pid[i])
                    #shutil.copyfile(pathImageDirectory + '/' + items[2], '/home/unath/medical_imaging_projects/ModelsGenesis/pytorch/test_data/pre/'+self.pre_pid[i]+'/'+items[2])
                    img = nib.load(pathImageDirectory + '/' + img_path).get_fdata()
                    if self.pre_pid[i] in self.patient_mri_pre:
                       self.patient_mri_pre[self.pre_pid[i]].append(img)
                    else:
                       self.patient_mri_pre[self.pre_pid[i]] = [img]

            for items in self.item_list:
                img_path = items[1]
                if img_path.find(self.post_pid[i])!=-1:
                    #if not os.path.exists('test_data/post/'+self.post_pid[i]):
                    #   os.mkdir('test_data/post/'+self.post_pid[i])
                    #shutil.copyfile(pathImageDirectory + '/' + items[2], '/home/unath/medical_imaging_projects/ModelsGenesis/pytorch/test_data/post/'+self.post_pid[i]+'/'+items[2])
                    img = nib.load(pathImageDirectory + '/' + img_path).get_fdata()
                    if self.post_pid[i] in self.patient_mri_post:
                       self.patient_mri_post[self.post_pid[i]].append(img)
                    else:
                       self.patient_mri_post[self.post_pid[i]] = [img]


        transformations_list = []
        #if self.train:
        #   transformations_list.append(transforms.RandomRotation(degrees=(-30, 30)))
        self.transformSequence = transforms.Compose(transformations_list)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        pre_pid = self.pre_pid[idx]
        post_pid = self.post_pid[idx]
        pre_images = None
        post_images = None
        label = self.label[idx]

        for i in range(len(self.patient_mri_pre[pre_pid])):
            img = self.patient_mri_pre[pre_pid][i]
            img = cv2.resize(img, (160, 160))
            img = self.transformSequence(torch.tensor(img))

            pad_length = 112 - img.shape[2]
            pad = (0, pad_length)
            if pad_length > 0:
               img = F.pad(img, pad, "constant", 0)

               x = img[None,:]
            else:
               x = img[None,:,:,:112]
            x = window(x)
            if pre_images==None:
               pre_images = x
            else:
               pre_images = torch.cat((pre_images, x))

        for i in range(len(self.patient_mri_post[post_pid])):
            img = self.patient_mri_post[post_pid][i]
            img = cv2.resize(img, (160, 160))
            img = self.transformSequence(torch.tensor(img))

            pad_length = 112 - img.shape[2]
            pad = (0, pad_length)
            if pad_length > 0:
               img = F.pad(img, pad, "constant", 0)

               x = img[None,:]
            else:
               x = img[None,:,:,:112]
            x = window(x)
            if post_images==None:
               post_images = x
            else:
               post_images = torch.cat((post_images, x))

        n_series = pre_images.shape[0]
        if n_series<7:
           pad = torch.zeros(7-n_series, 160, 160, 112)
           pre_images = torch.cat((pre_images, pad))

        n_series = post_images.shape[0]
        if n_series<7:
           pad = torch.zeros(7-n_series, 160, 160, 112)
           post_images = torch.cat((post_images, pad))

        return pre_images[:7], post_images[:7], label

def imageTransformation(img, test=False):
    dim = 128
    ct_scan = window(img)
    
    patient_image = ct_scan[None,:]

    # Remove maximum extent of the zero-background to make future crop more useful
    z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
    # Add 1 pixel in each side
    zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
    zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
    patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
    # default to 128, 128, 128
    if test:
        patient_image = pad_or_crop_image(patient_image, None, target_size=(dim, dim, dim), test=True)
    else:
        patient_image = pad_or_crop_image(patient_image, None, target_size=(dim, dim, dim))
    patient_image = patient_image.astype("float16")
    patient_image = torch.from_numpy(patient_image)
    return patient_image


class KrasClassificationDataset(Dataset):

    def __init__(self, pathDiretory, trainDataset=True, train=True, dim=(224, 224, 3), mean=0.4142, std=0.3993):
        self.pathDiretory = pathDiretory
        self.isTest = not train
        if trainDataset:
           self.path = self.pathDiretory + '/train/'
           self.seg_dir = '/scratch/unath/segmetation_kras/train/pred/'
        else:
           self.path = self.pathDiretory + '/test/'
           self.seg_dir = '/scratch/unath/segmetation_kras/val/pred/'

        self.seg_dir = '/scratch/unath/segmetation_kras/all/'
        self.patient = os.listdir(self.path)
        self.patientid = []
        self.label = []


    def __len__(self):
        return len(self.patient)

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __getitem__(self, idx):
        #print(self.patient[idx].shape, self.label[idx])
        img = self.load_nii(self.path + '/' + self.patient[idx])

        patient_image = imageTransformation(img, test=self.isTest)
        seg_pred_path = self.seg_dir + self.patient[idx].split('.')[0]+'.pt'
        seg_pred = torch.load(seg_pred_path, map_location='cpu')
        
        #print(patient_image.shape, seg_pred[0].shape) 
        patient_image = patient_image * seg_pred[0]
        
        if "w" in self.patient[idx].split('-')[1]:
            # print(self.patient[idx].split('-')[1])
            label = 0
        else:
            label = 1
        return patient_image, torch.tensor(label)


def findPartern(mylist, p):
    for index in mylist:
        if p in index:
           return index

class KrasClassificationDatasetV2(Dataset):

    def __init__(self, pathDiretory, train=True, train_set=['010-c','016-c','115-c','026-c'], test_set=['048-c', '097-c', '062-c']):
        self.pathDiretory = pathDiretory
        if train:
           self.path = self.pathDiretory + '/train'
           self.seg_dir = '/scratch/unath/segmetation_kras/train/pred/'
        else:
           self.path = self.pathDiretory + '/test'
           self.seg_dir = '/scratch/unath/segmetation_kras/val/pred/'

        self.seg_dir = '/scratch/unath/segmetation_kras/all/' 
        self.patient = os.listdir(self.path)
        self.patientid = []
        for index in self.patient:
            pid = index.split('-')[2] + '-c'
            if (pid in train_set) or (pid in test_set):
               pid_nc = findPartern(self.patient, index.split('-')[2] + '-nc')
               self.patientid.append([index, pid_nc])
            else:
               self.patientid.append([index])
        self.label = []


    def __len__(self):
        print(len(self.patientid))
        return len(self.patientid)

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __getitem__(self, idx):
        #print(self.patient[idx].shape, self.label[idx])
        
        img = self.load_nii(self.path + '/' + self.patientid[idx][0])
        
        if len(self.patientid[idx]) > 1:
           
           img2 = self.load_nii(self.path + '/' + self.patientid[idx][1])
           patient_image2 = imageTransformation(img2)
           seg_pred_path = self.seg_dir + self.patientid[idx][1].split('.')[0]+'.pt'
           seg_pred = torch.load(seg_pred_path, map_location='cpu')
           patient_image2 = patient_image2 * seg_pred[0]

        patient_image = imageTransformation(img)
        seg_pred_path = self.seg_dir + self.patientid[idx][0].split('.')[0]+'.pt'
        seg_pred = torch.load(seg_pred_path, map_location='cpu')
        patient_image = patient_image * seg_pred[0]

        #seg_pred_path = self.seg_dir + self.patient[idx].split('.')[0]+'.pt'
        #seg_pred = torch.load(seg_pred_path, map_location='cpu')

        #patient_image = patient_image * seg_pred[0]

        if "w" in self.patientid[idx][0].split('-')[1]:
            # print(self.patient[idx].split('-')[1])
            label = 0
        else:
            label = 1
        if len(self.patientid[idx]) > 1:
          return [patient_image, patient_image2], torch.tensor(label), torch.tensor(1)

        return [patient_image, patient_image], torch.tensor(label), torch.tensor(-1)

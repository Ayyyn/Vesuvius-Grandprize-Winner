import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger
import gc
import random

import numpy as np
import pandas as pd

# import wandb

from torch.utils.data import DataLoader

import pandas as pd
import os
import random
from contextlib import contextmanager
import cv2

import scipy as sp
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

import datetime
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
# from models.i3dallnl import InceptionI3d
import torch.nn as nn
import torch
# from warmup_scheduler import GradualWarmupScheduler
from scipy import ndimage
from resnetall import generate_model
import PIL.Image
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

from tap import Tap
class InferenceArgumentParser(Tap):
    parser = argparse.ArgumentParser(description="Process a value dynamically.")
    parser.add_argument('--val_segment_id', help="Segment id for inference")
    parser.add_argument('--segment_path', help="Path of segment for inference")
    parser.add_argument('--model_path', help="Model path")
    parser.add_argument('--out_path', help="inference output path")
    # parser.add_argument('--model_name', help="name of model")

    #example command with arguments
    #python inference_timesformer.py --segment_id 20230701020044 --segment_path /path/to/segment --model_path /path/to/model --out_path /path/to/output
    # Parse the arguments
    args = parser.parse_args()

    val_segment_id:str =args.val_segment_id
    segment_path:str=args.segment_path
    model_path:str=args.model_path
    out_path:str=args.out_path
    # model_name:str=args.model_name

    print(f"{val_segment_id=}")
    print(f"{segment_path=}")
    print(f"{model_path=}")
    print(f"{out_path=}")

args = InferenceArgumentParser().parse_args()

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = './'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    # comp_dataset_path = f'./'
    comp_dataset_path = f'/content/t1/vesuvius_model/training/'
    
    exp_name = 'pretraining_all'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    backbone='resnet3d'
    in_chans = 30 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 256
    tile_size = 256
    # stride = tile_size // 8
    stride = 32

    train_batch_size = 16 # 32
    valid_batch_size = train_batch_size
    # use_amp = True

    # scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 10 # 30

    # adamW warmupあり
    # warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    # lr = 1e-4 / warmup_factor
    lr = 3e-5
    # ============== fold =============
    valid_id = '20230531121653'

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    # metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    # pretrained = True
    # inf_weight = 'best'  # 'best'

    # min_lr = 1e-6
    # weight_decay = 1e-6
    # max_grad_norm = 100

    # print_freq = 50
    num_workers = 16

    seed = 130697

    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'{comp_dataset_path}/outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.6),

        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    rotate = A.Compose([A.Rotate(5,p=1)])
# def init_logger(log_file):
#     from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
#     logger = getLogger(__name__)
#     logger.setLevel(INFO)
#     handler1 = StreamHandler()
#     handler1.setFormatter(Formatter("%(message)s"))
#     handler2 = FileHandler(filename=log_file)
#     handler2.setFormatter(Formatter("%(message)s"))
#     logger.addHandler(handler1)
#     logger.addHandler(handler2)
#     return logger

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image_mask(fragment_id,start_idx=60,end_idx=90):
    fragment_id_ = fragment_id.split("_")[0]
    images = []
    if fragment_id==args.val_segment_id:
      start_idx = 17
      end_idx = 47
    # idxs = range(65)
    # mid = 65 // 2
    # start = mid - CFG.in_chans // 2
    # end = mid + CFG.in_chans // 2
    idxs = range(start_idx, end_idx)

    for i in idxs:
        if os.path.exists(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif"):
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif", 0)
        elif os.path.exists(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.jpg"):
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.jpg", 0)
        elif os.path.exists(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:03}.tif"):
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:03}.tif", 0)
        elif os.path.exists(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:03}.jpg"):
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:03}.jpg", 0)

        if i==27 or i==70:
            print(f"{fragment_id}: {i} layer found")
            print(f"before padding: {fragment_id} layer shape= {image.shape}")

        # width_m1 = int(image.shape[1]//(30/14.5))
        # width_m2 = int(image.shape[1]//(30/19.5))
        # image=image[:, image.shape[1]//4:]

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = ndimage.median_filter(image, size=5)
        
        # image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        if 'frag' in fragment_id:
            image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        image=np.clip(image,0,200)
        if fragment_id_=='20230827161846':
            image=cv2.flip(image,0)
        # if fragment_id != args.val_segment_id:
        #     image=image[1361:3038, 2272:3513]
        images.append(image)
    images = np.stack(images, axis=2)

    #********************************label********************************
    # if fragment_id_ in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']:
    # if fragment_id_ in ['20240820133348','20240814122007', '20240814125124']:
    #     images=images[:,:,::-1] #reverse the order of the image slices
    if fragment_id_ in ['20231022170901','20231022170900']:
        mask = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_inklabels.tiff", 0)
    else:
        if os.path.exists(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_inklabels.png"):
            #print the path of the mask
            print(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_inklabels.png")
            print(f"{fragment_id}: inklabel found")
            mask = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_inklabels.png", 0)
        else:
            print(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_inklabels.png")
            print(f"{fragment_id}: inklabel not found")

    # mask = mask[:, width_m1:width_m2]
    # if fragment_id != args.val_segment_id: 
    #     mask = mask[1361:3038, 2272:3513]  
    # mask = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_inklabels.png", 0)
    # if fragment_id != args.val_segment_id:
    #   mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)


    #***********************************fragment mask********************************
    if os.path.exists(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_mask.png"):
        print(f"{fragment_id}: mask found")
    else:
        print(f"{fragment_id}: mask not found")

    fragment_mask=cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_mask.png", 0)
    if fragment_id_=='20230827161846':
        fragment_mask=cv2.flip(fragment_mask,0)

    # fragment_mask = fragment_mask[:, fragment_mask.shape[1]//4:]
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

    #******************
    kernel = np.ones((16,16),np.uint8)
    if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)

    #normalize the mask
    mask = mask.astype('float32')
    mask/=255
    # if fragment_id != args.val_segment_id:
        # fragment_mask = fragment_mask[1361:3038, 2272:3513]

    print(f"{fragment_id} layer shape= {images.shape}")
    print(f"{fragment_id} label shape: {mask.shape}")
    print(f"{fragment_id} mask shape: {fragment_mask.shape}")
    assert images.shape[0]==mask.shape[0]
    return images, mask,fragment_mask

def get_train_valid_dataset(segment_list):
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []
    # for fragment_id in [args.valid_segment_id,'20231106155350','20231005123336','20230820203112','20230620230619','20230826170124','20230702185753','20230522215721','20230531193658','20230520175435','20230903193206','20230902141231','20231007101615','20230929220924','recto','verso','20231016151000','20231012184423','20231031143850']:  
    # for fragment_id in ['20230820203112','20231005123333']:
    # for fragment_id in ['20230522181603','20230702185752','20230827161847','20230909121925','20230905134255','20230904135535']:
    for fragment_id in segment_list:
        print('reading ',fragment_id)
        # try:
        #     image, mask,fragment_mask = read_image_mask(fragment_id)
        #     print("******{fragment_id}******")
        #     print(f"{image.shape=}")
        #     print(f"{mask.shape=}")
        #     print(f"{fragment_mask.shape=}")
        # except:
        #     print(f"couldnt load {fragment_id}!")
        image, mask,fragment_mask = read_image_mask(fragment_id)
        # print("******{fragment_id}******")
        # print(f"{image.shape=}")
        # print(f"{mask.shape=}")
        # print(f"{fragment_mask.shape=}")
        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
        windows_dict={}
        for a in y1_list:
            for b in x1_list:
                for yi in range(0,CFG.tile_size,CFG.size):
                    for xi in range(0,CFG.tile_size,CFG.size):
                        y1=a+yi
                        x1=b+xi
                        y2=y1+CFG.size
                        x2=x1+CFG.size
                        if fragment_id!=CFG.valid_id:
                            if not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]<0.01):
                                if not np.any(fragment_mask[a:a+ CFG.tile_size, b:b + CFG.tile_size]==0):
                                    if (y1,y2,x1,x2) not in windows_dict:
                                        train_images.append(image[y1:y2, x1:x2])
                                        
                                        train_masks.append(mask[y1:y2, x1:x2, None])
                                        assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                                        windows_dict[(y1,y2,x1,x2)]='1'
                        if fragment_id==CFG.valid_id:
                            if not np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size]==0):
                                    valid_images.append(image[y1:y2, x1:x2])
                                    valid_masks.append(mask[y1:y2, x1:x2, None])

                                    valid_xyxys.append([x1, y1, x2, y2])
                                    assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug

class CustomDataset(Dataset):
    def __init__(self, images ,cfg,xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        
        self.transform = transform
        self.xyxys=xyxys
        self.rotate=CFG.rotate
    def __len__(self):
        return len(self.images)
    def cubeTranslate(self,y):
        x=np.random.uniform(0,1,4).reshape(2,2)
        x[x<.4]=0
        x[x>.633]=2
        x[(x>.4)&(x<.633)]=1
        mask=cv2.resize(x, (x.shape[1]*64,x.shape[0]*64), interpolation = cv2.INTER_AREA)

        
        x=np.zeros((self.cfg.size,self.cfg.size,self.cfg.in_chans)).astype(np.uint8)
        for i in range(3):
            x=np.where(np.repeat((mask==0).reshape(self.cfg.size,self.cfg.size,1), self.cfg.in_chans, axis=2),y[:,:,i:self.cfg.in_chans+i],x)
        return x
    def fourth_augment(self,image):
        image_tmp = np.zeros_like(image)
        # cropping_num = random.randint(24, 30)
        cropping_num = random.randint(CFG.in_chans-6, CFG.in_chans)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            #3d rotate
            image=image.transpose(2,1,0)#(c,w,h)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,h,w)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,w,h)
            image=image.transpose(2,1,0)#(h,w,c)

            image=self.fourth_augment(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label
class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image,xy



# from resnetall import generate_model
# def init_weights(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask



class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=256,enc='',with_norm=False,total_steps=780):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        self.backbone = generate_model(model_depth=50, n_input_channels=1,forward_features=True,n_classes=1139)
        # state_dict=torch.load('./r3d101_KM_200ep.pth')["state_dict"]
        # state_dict=torch.load(args.model_path)["state_dict"]
        # conv1_weight = state_dict['conv1.weight']
        # state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        
        # conv1_weight = state_dict['backbone.conv1.weight']
        # state_dict['backbone.conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        
        # self.backbone.load_state_dict(state_dict,strict=False)
        #freeze the convolution layers' parameters in the backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad_ = False

        # self.backbone.requires_grad_(False)

        # self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=True)
        # self.backbone.load_state_dict(torch.load('./pretraining_i3d_epoch=3.pt'),strict=False)
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        
        return pred_mask
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        # wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

    def configure_optimizers(self):
        # optimizer = AdamW(self.parameters(), lr=CFG.lr)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr = CFG.lr)
        scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4,pct_start=0.15, steps_per_epoch=self.hparams.total_steps, epochs=50,final_div_factor=1e2)

        # scheduler = get_scheduler(CFG, optimizer)
        return [optimizer],[scheduler]

   

if __name__ == '__main__':

    fragment_id = args.val_segment_id
    # valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)
    # # valid_mask_gt=cv2.resize(valid_mask_gt,(valid_mask_gt.shape[1]//2,valid_mask_gt.shape[0]//2),cv2.INTER_AREA)
    # pred_shape=valid_mask_gt.shape
    torch.set_float32_matmul_precision('medium')

    fragments=[args.val_segment_id]
    enc_i,enc,fold=0,'i3d',0
    for fid in fragments:
        CFG.valid_id=fid
        fragment_id = CFG.valid_id
        run_slug=f'training_scrolls_valid={fragment_id}_{CFG.size}x{CFG.size}_submissionlabels_ftune'

        valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)

        pred_shape=valid_mask_gt.shape
        segment_list = ['20231111135345', '20230531121653']
        train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset(segment_list)
        print(len(train_images))
        valid_xyxys = np.stack(valid_xyxys)
        train_dataset = CustomDataset(
            train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
        valid_dataset = CustomDataset(
            valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

        train_loader = DataLoader(train_dataset,
                                    batch_size=CFG.train_batch_size,
                                    shuffle=True,
                                    num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                    )
        valid_loader = DataLoader(valid_dataset,
                                    batch_size=CFG.valid_batch_size,
                                    shuffle=False,
                                    num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

        # wandb_logger = WandbLogger(project="vesivus",name=run_slug+f'{enc}_finetune')
        norm=fold==1
        model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size,total_steps=len(train_loader))

        #for finetuning
        model.load_state_dict(torch.load(args.model_path)["state_dict"])

        for module in model.modules():
        # print(module)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

        # Freeze only the 'backbone.conv1.weight' layer
        layers_conv_layers_list = ["backbone.conv1.weight", 
                                   "backbone.layer1.0.conv1.weight", "backbone.layer1.0.conv2.weight", "backbone.layer1.0.conv3.weight", 
                                   "backbone.layer1.1.conv1.weight", "backbone.layer1.1.conv2.weight", "backbone.layer1.1.conv3.weight", 
                                   "backbone.layer1.2.conv1.weight", "backbone.layer1.2.conv2.weight", "backbone.layer1.2.conv3.weight",
                                  #  "backbone.layer2.0.conv1.weight", "backbone.layer2.0.conv2.weight", "backbone.layer2.0.conv3.weight",
                                  #  "backbone.layer2.1.conv1.weight", "backbone.layer2.1.conv2.weight", "backbone.layer2.1.conv3.weight",
                                  #  "backbone.layer2.2.conv1.weight", "backbone.layer2.2.conv2.weight", "backbone.layer2.2.conv3.weight",
                                   ]
        for name, param in model.named_parameters():
            if name in layers_conv_layers_list:
                param.requires_grad = False
                print(f"Froze layer: {name}")
            # else:
            #     print(f"Did not freeze: {name}")

        
        print('FOLD : ',fold)
        # wandb_logger.watch(model, log="all", log_freq=100)
        # multiplicative = lambda epoch: 0.9

        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val/total_loss',  # Monitor validation MSE loss
            patience=4,              # Stop training after 4 epochs without improvement
            verbose=True,            # Enable logging
            mode='min',              # We want to minimize the validation loss
            check_on_train_epoch_end=False  # Check metric only at validation epoch end
        )

        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="gpu",
            # devices=8,
            # check_val_every_n_epoch=5,
            # logger=wandb_logger,
            default_root_dir="./models",
            accumulate_grad_batches=1,
            precision='16-mixed',
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            strategy='ddp_find_unused_parameters_true',
            callbacks=[ModelCheckpoint(
                filename=f'finetuned_7_45_exposedfrag_og_resnet_wild14_frags_extended4_18chans_{fid}_{fold}_fr_{enc}'+'{epoch}',
                dirpath=CFG.model_dir,
                monitor='train/total_loss',
                mode='min',
                save_top_k=-1,
                every_n_epochs = 1),
                # early_stopping
                        ],
        )

        #https://www.perplexity.ai/search/what-are-the-keys-stored-in-a-2mbWRfpAQF.S9sS.tK4ifw
        trainer.fit(model=model, train_dataloaders=train_loader)
        del train_images,train_loader,train_masks,valid_loader,model
        torch.cuda.empty_cache()
        gc.collect()
        #wandb.finish()
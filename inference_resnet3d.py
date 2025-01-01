import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler
import argparse
import random
import numpy as np
import pandas as pd
import gc

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging,BackboneFinetuning

import scipy.stats as st
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
from torch.utils.data import DataLoader
import cv2
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image
from PIL import Image
from resnetall import generate_model
PIL.Image.MAX_IMAGE_PIXELS = 933120000
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tap import Tap
class InferenceArgumentParser(Tap):
    parser = argparse.ArgumentParser(description="Process a value dynamically.")
    parser.add_argument('--segment_id', help="Segment id for inference")
    parser.add_argument('--segment_path', help="Path of segment for inference")
    parser.add_argument('--model_path', help="Model path")
    parser.add_argument('--out_path', help="inference output path")
    parser.add_argument('--model_name', help="model name")

    #example command with arguments
    #python inference_timesformer.py --segment_id 20230701020044 --segment_path /path/to/segment --model_path /path/to/model --out_path /path/to/output
    # Parse the arguments
    args = parser.parse_args()

    segment_id: list[str] =args.segment_id
    segment_path:str=args.segment_path
    model_path:str=args.model_path
    out_path:str=args.out_path
    model_name:str=args.model_name

    print(f"{segment_id=}")
    print(f"{segment_path=}")
    print(f"{model_path=}")
    print(f"{out_path=}")
    print(f"{model_name=}")

args = InferenceArgumentParser().parse_args()
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = ''
    
    exp_name = 'pretraining_all'
    # ============== model cfg =============
    # model_name = 'Unet'
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'

    in_chans = 30 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 64
    tile_size = 64
    # stride = tile_size // 3
    stride = 16

    train_batch_size = 256 # 32
    valid_batch_size = 1024
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 50 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-4 / warmup_factor
    min_lr = 1e-6
    num_workers = 16
    seed = 42
    # ============== augmentation =============
    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
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

# def make_dirs(cfg):
#     for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
#         os.makedirs(dir, exist_ok=True)
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def read_image_mask(fragment_id,start_idx=18,end_idx=42,rotation=0):
    images = []
    # mid = 65 // 2
    # start = mid - CFG.in_chans // 2
    # end = mid + CFG.in_chans // 2
    idxs = range(start_idx, end_idx)
    for i in idxs:
        image = cv2.imread(f"{args.segment_path}/{fragment_id}/layers/{i:03}.tif", 0)
        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = ndimage.median_filter(image, size=5)
        image=np.clip(image,0,200)
        # image = cv2.flip(image, 0)
        images.append(image)
    images = np.stack(images, axis=2)
    if fragment_id in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']:
        images=images[:,:,::-1]
    print("Images' stack created")

    fragment_mask=None
    if os.path.exists(f'{args.segment_path}/{fragment_id}/{fragment_id}_mask.png'):
        fragment_mask=cv2.imread(CFG.comp_dataset_path + f"{args.segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

    return images,fragment_mask

def get_img_splits(fragment_id,s,e,rotation=0):
    images = []
    xyxys = []
    image,fragment_mask = read_image_mask(fragment_id,s,e,rotation)
    print("image mask craeted of shape: ",image.shape)
    print("fragment mask created of shape: ",fragment_mask.shape)
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])
    test_dataset = CustomDatasetTest(images,np.stack(xyxys), CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean= [0] * CFG.in_chans,
            std= [1] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
                              )
    return test_loader, np.stack(xyxys),(image.shape[0],image.shape[1]),fragment_mask

def get_transforms(data, cfg):
    if data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

class CustomDataset(Dataset):
    def __init__(self, images ,cfg,xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.xyxys=xyxys
        self.kernel=gkern(64,2)
        self.kernel/=(self.kernel.max() + 1e-8)
        self.kernel=torch.FloatTensor(self.kernel)
    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            offset=4
            image=image[:,:,offset:offset+self.cfg.in_chans]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label= torch.mul(self.kernel,data['mask'])
                label = label.mean().type(torch.float32)

            return image, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            # offset=random.choice([0,1,2,3,4])
            offset=4
            image=image[:,:,offset:offset+self.cfg.in_chans]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label= torch.mul(self.kernel,data['mask'])
                label = label.mean().type(torch.float32)
            
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
        # for l in self.convs:
        #     for m in l._modules:
        #         init_weights(m)
    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask



from collections import OrderedDict
class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=256,enc='',with_norm=False,total_steps=780):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        # if self.hparams.enc=='resnest101':
        self.backbone = generate_model(model_depth=50, n_input_channels=1,forward_features=True,n_classes=1139)
        # state_dict=torch.load('./r3d101_KM_200ep.pth')["state_dict"]
        state_dict=torch.load(args.model_path)["state_dict"]
        # conv1_weight = state_dict['conv1.weight']
        # state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.backbone.load_state_dict(state_dict,strict=False)

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
        self.log("train/Arcface_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
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

        self.log("val/MSE_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        # wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=CFG.lr)    
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer]

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

def predict_fn(test_loader, model, device, test_xyxys,pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    kernel=gkern(CFG.size,1)
    kernel=kernel/kernel.max()
    model.eval()

    for step, (images,xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)
        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy(),kernel)
            # mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    print(f"Zeros in mask_count: {np.sum(mask_count == 0)} out of {mask_count.size}")
    mask_pred /= (mask_count + 1e-8)

    return mask_pred
    # return losses.avg,[]
if __name__ == "__main__":
    model=RegressionPLModel.load_from_checkpoint(args.model_path,strict=False,enc='resnest101')
    model.to(device)
    # model.cuda()
    model = torch.compile(model)
    model.eval()

    for fragment_id in args.segment_id:
        print(f"Processing {fragment_id=}")
        i = 20
        if os.path.exists(f"{args.segment_path}/{fragment_id}/layers/070.tif"):
            preds=[]
            for r in [0]:
                for i in [60]:
                    start_f=i
                    print(f'{start_f=}')
                    end_f=start_f+CFG.in_chans
                    print(f"{end_f=}")
                    test_loader,test_xyxz,test_shape,fragment_mask=get_img_splits(fragment_id,start_f,end_f,r)
                    mask_pred= predict_fn(test_loader, model, device, test_xyxz,test_shape)
                    mask_pred=np.clip(np.nan_to_num(mask_pred),a_min=0,a_max=1)
                    mask_pred/= (mask_pred.max() + 1e-8)
                    mask_pred = (mask_pred * 255).astype(np.uint8)

                    preds.append(mask_pred)
            
            os.makedirs(f'{args.out_path}', exist_ok=True)
            img = Image.fromarray(preds[0])
            img.save(f"{args.out_path}/{fragment_id}-mname_{args.model_name}-s_{start_f}-e_{end_f}-stride_{CFG.stride}.png")
            gc.collect()
        else:
            print("******Path doesn't exist******")
    del mask_pred,test_loader,model
    torch.cuda.empty_cache()
    gc.collect()
    #wandb.finish()
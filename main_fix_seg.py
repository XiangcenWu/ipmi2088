import re
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

import monai
from monai.networks.nets.unet import UNet
from monai.losses import DiceLoss

from model import SelectionNet
from train import *




import argparse
#################################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('device', type=str, help='device to calculate')
parser.add_argument('shuffle', type=str2bool, help='Shuffle the training data')
parser.add_argument('num_sequence', type=int, help='Sequence Length')
parser.add_argument('nickname', type=str, help='saved stuff nickname')
args = parser.parse_args()
print(args.shuffle, args.nickname)

device = args.device
##############################


root_dir = "./data/malignant"
regex = re.compile(r"\d+")

def path_list_ranking(path_list):
    return sorted(path_list, key=lambda i: int(regex.findall(i)[0]))

def data_path_list(root_dir):
    path = Path(root_dir)
    all_paths = [str(image_path) for image_path in path.glob("*.png")]
    imgs_list = [img_path for img_path in all_paths if "_mask" not in img_path]
    labels_list = [label_path for label_path in all_paths if "_mask" in label_path]
    return path_list_ranking(imgs_list), path_list_ranking(labels_list)


imgs_list, labels_list = data_path_list(root_dir)


meta_train_list, meta_val_list, meta_train_label_list, meta_val_label_list = train_test_split(imgs_list, labels_list, test_size=120, train_size=90, random_state=25)
meta_val_list, holdout_list, meta_val_label_list, holdout_label_list = train_test_split(meta_val_list, meta_val_label_list, test_size=30, train_size=90, random_state=25)

print(holdout_list[12], holdout_label_list[12])



class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, images_list :list, 
                 labels_list :list, 
                 img_transform :T=None ,
                 label_transform :T=None):
        super().__init__()
        
        assert len(images_list) == len(labels_list) # images labels have numbers doesn't match
        self.images_list = images_list
        self.labels_list = labels_list
        self.img_transform = img_transform
        self.label_transform = label_transform
        
        
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        image = Image.open(self.images_list[index])
        label = Image.open(self.labels_list[index])
        
        return self.img_transform(image), self.label_transform(label)

    
    
basic_transform = T.Compose([
    T.Resize([324,324]),
#     T.Grayscale(1),
#     T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 5)),
    T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

basic_label_transform = T.Compose([
    T.Resize([324,324]),
    T.ToTensor()
])
Seg_dataset = Dataset(meta_train_list, meta_train_label_list, basic_transform, basic_label_transform)
Meta_val_dataset = Dataset(meta_val_list, meta_val_label_list, basic_transform, basic_label_transform)
Holdout_test_dataset = Dataset(holdout_list, holdout_label_list, basic_transform, basic_label_transform)

print(Meta_val_dataset)

Meta_val_loader = DataLoader(Meta_val_dataset, batch_size=args.num_sequence, shuffle=args.shuffle, drop_last=True)
test_loader = DataLoader(Holdout_test_dataset, batch_size=args.num_sequence, shuffle=args.shuffle, drop_last=True)

Seg_model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 1, 1),
    num_res_units=3,
    dropout = 0.2
)
Seg_model.load_state_dict(torch.load("/home/xiangcen/ipmi2088/seg_fix.pt", map_location=device))
Seg_model.to(device)
Seg_model.eval()

Sel_model = SelectionNet(
    DS_dim_list=[3, 64, 256, 512, 1024, 2048], 
    num_resblock=2, 
    transformer_input_dim=2048, 
    num_head=8,
    dropout=0.1,
    num_transformer=6
)

Sel_model.to(device)

optimizer_sel = torch.optim.Adam(Sel_model.parameters(), lr = 1e-5)


dummy_list = []
for b in range(3000):
    sel_loss = train_sel_net_baseon_seg_net(Sel_model, Seg_model, Meta_val_loader, optimizer_sel, device=device)
    sel_loss_test = eval_sel_net_baseon_seg_net(Sel_model, Seg_model, test_loader, device=device)

    save = torch.tensor([sel_loss, sel_loss_test])
    print(save)
    dummy_list.append(save)
    torch.save(torch.stack(dummy_list), './saved_value_' + args.nickname + '.pt')

    torch.save(Sel_model.state_dict(), './sel_' + args.nickname + '.pt')

# -*- coding: utf-8 -*-
"""test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17bR1sEMAA7l68cS_QMha6LrbDKmCDHeb
"""

import warnings
warnings.filterwarnings("ignore")

import ImageEncoder
from convolutional_occupancy_networks.src.conv_onet.models.decoder import LocalDecoder
from convolutional_occupancy_networks.src.conv_onet import generation
from convolutional_occupancy_networks.src.common import compute_iou
from convolutional_occupancy_networks.src import data as srcData

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms.functional as transform
import torchvision as tv
from torch import distributions as dist
from torch.utils import data

import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml

class ImageOccupancyNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.encoder = ImageEncoder.LocalImageEncoder(device).to(device)
        self.decoder = LocalDecoder(dim=3,sample_mode="bilinear", c_dim=64, hidden_size=64, n_blocks=5).to(device)
        self._device = device

    def forward(self, p, inputs):
       c = self.encode_inputs(inputs)
       p_r = self.decode(p, c)
       return p_r

    def encode_inputs(self, inputs):
        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            print("empty")
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c):
        logits = self.decoder(p, c)
        p_r = dist.Bernoulli(logits=logits)
        return p_r
        
    def to(self, device):
        model = super().to(device)
        model._device = device
        return model

class Trainer():
        def __init__(self, model, optimizer, device):
            self.model = model
            self.optimizer = optimizer
            self.device = device

        def train_step(self, batch):
            #print("Entered train_step")
            self.model.train()
            self.optimizer.zero_grad()
            loss =  self.calculate_loss(batch)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        
        def calculate_loss (self, data):
            #print("calc loss")
            device = self.device
            p = data.get('points').to(device)
            occ = data.get('points.occ').to(device)
            inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

            logits = self.model(p, inputs).logits
            
            loss_i = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
            loss = loss_i.sum(-1).mean()

            return loss

        def eval_step(self, data):
            self.model.eval()

            device = self.device
            eval_dict = {}

            points = data.get('points').to(device)
            inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
            occ_iou = data.get('points.occ').to(device)
            #points_iou = data.get('points_iou').to(device)
            #occ_iou = data.get('points_iou.occ').to(device)

            with torch.no_grad():
                p_out = self.model(points, inputs)


            logits = p_out.logits
            
            loss_i = F.binary_cross_entropy_with_logits(logits, occ_iou, reduction='none')
            loss = loss_i.sum(-1).mean()


            occ_iou_np = (occ_iou >= 0.5).tolist()
            occ_iou_hat_np = (p_out.probs >= 0.5).tolist()

            iou = compute_iou(np.array(occ_iou_np), np.array(occ_iou_hat_np)).mean()
            eval_dict['iou'] = iou
            eval_dict['loss'] = loss.item()

            return eval_dict

        def evaluate(self, val_loader):
            ious = []
            losses = []

            for data in tqdm(val_loader):
                iou = self.eval_step(data)
                ious.append(iou["iou"])
                losses.append(iou["loss"])

            return np.mean(ious), np.mean(losses)

class CustomDataset(Dataset):
    def __init__(self, img_dir, split):
        self.img_dir = img_dir
        self.split = split
        self.models = []

        self.metadataFile = img_dir + "metadata.yaml"
        with open(self.metadataFile, 'r') as f:
                self.metadata = yaml.load(f)

        
        for category in self.metadata:
            folder = img_dir + str(category)
            split_file = folder + "\\" +  split + '.lst'
            with open(split_file, 'r') as f:
                catModels = f.read().split('\n')
            
            if '' in catModels:
                catModels.remove('')

        
            self.models.extend([str(category) + "\\" + model for model in catModels if os.path.exists(folder + "\\" + model)])

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        img_path = self.img_dir +  self.models[idx] +  "\\imageSmall.png"
        points_path = self.img_dir +  self.models[idx] +  "\\points.npz"

        image = tv.io.read_image(img_path)
        points_dict = np.load(points_path)
        points = points_dict['points']

        points = points.astype(np.float64)
        points += 1e-4 * np.random.randn(*points.shape)
        points[:, 0] = points[:, 0]*np.cos(np.pi) + points[:, 2]*np.sin(np.pi);
        points[:, 2] = points[:, 2]*np.cos(np.pi) - points[:, 0]*np.sin(np.pi);

        occupancies = points_dict['occupancies']
        occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float64)
        #norm  = tv.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))

        points_transform = srcData.SubsamplePoints(8192)
        subsampledPoints = points_transform({None : points, "occ" : occupancies})

        returnVal = {
            "inputs": image.float(),
            "points": subsampledPoints[None],
            'points.occ': subsampledPoints["occ"],
            "points_iou" : points,
            "points_iou.occ" : occupancies
        }
        return returnVal

    def collate_remove_none(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return data.dataloader.default_collate(batch)

from convolutional_occupancy_networks.src.conv_onet import generation

def Generate(epochindex):
    generator3 = generation.Generator3D(
            network,
            device=device,
            threshold=0.4,
            resolution0=32,
            upsampling_steps=3,
            sample=False,
            refinement_step=0,
            input_type = None,
            padding=0.1,
            vol_info = None,
            vol_bound = None,
        )
    
    generator4 = generation.Generator3D(
            network,
            device=device,
            threshold=0.5,
            resolution0=32,
            upsampling_steps=3,
            sample=False,
            refinement_step=0,
            input_type = None,
            padding=0.1,
            vol_info = None,
            vol_bound = None,
        )
    
    VisFolder = "E:\\Szakdoga\\vis3\\"
    
    print("\nGenerating visualisations\n")
    for categoryFolder in tqdm(os.listdir(VisFolder)):
        if os.path.isdir(VisFolder + categoryFolder):
            for modelFolder in os.listdir(VisFolder + categoryFolder):
                img_path = VisFolder +  categoryFolder + "\\" + modelFolder + "\\imageSmall.png"
                image = tv.io.read_image(img_path)
                data = {
                    "inputs": image.float(),
                }
                out3 = generator3.generate_mesh(data)
                out4 = generator4.generate_mesh(data)
                try:
                    mesh3, stats_dict = out3
                    mesh4, stats_dict = out4
                except TypeError:
                    mesh3, stats_dict = out3, {}
                    mesh4, stats_dict = out4, {}
                fileName = "\\epoch" + str(epochindex) + "Resolution"
                mesh3.export(VisFolder +  categoryFolder + "\\" + modelFolder + fileName + str(3) + ".obj")
                mesh4.export(VisFolder +  categoryFolder + "\\" + modelFolder + fileName + str(4) + ".obj")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
network = ImageOccupancyNetwork(device)
optimizer = optim.AdamW(network.parameters(), lr=1e-4, )
network.load_state_dict(torch.load("E:\\Szakdoga\\Model3-Save90.pt"))
optimizer.load_state_dict(torch.load("E:\\Szakdoga\\Optimizer3-Save90.pt"))

#pytorch_total_params = sum(p.numel() for p in network.parameters())
#print(pytorch_total_params)

if __name__ == '__main__':
    train_data = CustomDataset("E:\\Szakdoga\\Data\\preprocessed\\ShapeNet\\", "train")
    val_data = CustomDataset("E:\\Szakdoga\\Data\\preprocessed\\ShapeNet\\", "test")
    train_loader = torch.utils.data.DataLoader(train_data,  batch_size=16, shuffle=True, num_workers=8, prefetch_factor = 8)
    val_loader = torch.utils.data.DataLoader(val_data,  batch_size=1, shuffle=False, num_workers=4)

    trainer = Trainer(network, optimizer, device)
   
    EvalIOU = []
    TrainingLoss = []

    for epoch in tqdm(range(100, 115)):
        trainLosses = []
        for batch in tqdm(train_loader):
            trainLosses.append(trainer.train_step(batch))
    
    
        evalIOU, evalLoss = trainer.evaluate(val_loader)
        print("\n\nEpoch ", epoch, " Eval IOU: ", evalIOU)
        print("Epoch ", epoch, " Training Loss: ", np.mean(trainLosses)/4, "\n\n")

        EvalIOU.append(evalIOU)
        TrainingLoss.append(np.mean(trainLosses)/4)

        plt.clf()
        plt.plot(TrainingLoss, label = "Training loss")
        plt.legend()
        plt.savefig("E:\\Szakdoga\\Losses\\Epoch " + str(epoch) + "Losses.png")

        plt.clf()
        plt.plot(EvalIOU, label = "Evalution IOU")
        plt.legend()
        plt.savefig("E:\\Szakdoga\\Losses\\Epoch " + str(epoch) + "IOU.png")

        if(epoch % 3 == 0):
            Generate(epoch)

        if(epoch % 5 == 0):
            torch.save(network.state_dict(), "E:\\Szakdoga\\Model3-Save" + str(epoch) + ".pt")
            torch.save(optimizer.state_dict(), "E:\\Szakdoga\\Optimizer3-Save" + str(epoch) + ".pt")
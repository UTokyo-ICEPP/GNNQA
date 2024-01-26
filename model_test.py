import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import math

import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

torch.manual_seed(0)

import os, sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )


print('cuda_device : ', cuda_device)

#from modules.TrackDataloader import  OneTrackDataset, collate_graphs
from modules.TrackDataloader_skip_test import  OneTrackDataset, collate_graphs

from modules.dynamic_graph_PhaseIII import Dynamic_Graph_Model

path = '/data/wachan/PhaseIII_data/TrackML_All/'

data_set_train = OneTrackDataset(path, num_start=2395, num_end=2400)

print(data_set_train)

#num_workers = number of GPU core, max = 56 in iutgpu02
train_loader = DataLoader(data_set_train, batch_size=1, shuffle=True,collate_fn=collate_graphs, num_workers=0)
#train_loader = DataLoader(data_set_train, batch_size=1, shuffle=True,collate_fn=collate_graphs, num_workers=56,pin_memory=True)
print ("Train Loader",len(train_loader))

model = Dynamic_Graph_Model(nlayers=3, nclass=1, nodes=16, device=cuda_device)
model.to(cuda_device)

print( 'Model Cuda : ', next(model.parameters()).is_cuda )

#print ('TRAIN ',data_set_train[3])

#data = next(iter(train_loader))#data_set_train[3]
#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.BCEWithLogitsLoss()

#gr_list, label = data
print ("=======Model Test=======")

missing_evt = pd.read_csv(path + 'Sample_missing.csv').evt.tolist()
print (missing_evt)

for batch, (gr_list, labelpair) in enumerate(train_loader): #Looping events
    if not gr_list:
        print ("gr_list empty, skipping...")
    elif gr_list:

        print('N graphs : ', len(gr_list))
        print ("gr_list: ",gr_list)
        print ("labelpair: ",labelpair)
        graph_list = []

        for ig in gr_list:
            print ("IG:",ig)

            for i in range(len(ig)):

                ig[i] = ig[i].to(cuda_device)

                graph_list.append(ig[i])

        print ("graph_list: ",graph_list)
        print ("N in graph_list: ",len(graph_list))

        bg = dgl.batch(graph_list)

        bg = bg.to(cuda_device)

        print ("bg.ndata['x']:",bg.ndata['x'])
        print ("bg.ndata['en']:",bg.ndata['en'])
        pred_label,edge_array,node_array = model(bg)
        # edge_array ... src:(0,1,2,...) dst:(1,2,3,...)
        # node_array ... 22905.,  30421.,  37893.,  ...
        print("pred_label:",pred_label.shape)
        print("edge_array:",edge_array[0][0].shape)
        label_list = torch.empty((0,1),dtype=torch.float,device=cuda_device)

        for (edge_pair, node_list, labelpair_list_all) in zip(edge_array, node_array, labelpair):

            edge_list = []
            for idx_ep in range(len(edge_pair[0])):
                idx_ep0 = edge_pair[0][idx_ep].item()
                idx_ep1 = edge_pair[1][idx_ep].item()

                edge_list.append([node_list[idx_ep0].item(),node_list[idx_ep1].item()])

            labelpair_list = []
            for lps in range(len(labelpair_list_all)):
                for l in labelpair_list_all[lps]:
                    labelpair_list.append(l)

            for e_HID in edge_list:
                ismatch = 0
                for lp in labelpair_list:
                    if int(lp[0]) == int(e_HID[0]) and int(lp[1]) == int(e_HID[1]):
                        ismatch = 1
                        print("A ",lp[0],lp[1],e_HID[0],e_HID[1])

                    else:
                        ismatch = 0

                    if ismatch == 1:
                        break


                if ismatch == 1:
                    label_list = torch.cat([label_list, torch.ones((1,1),dtype=torch.float,device=cuda_device)], dim=0)
                else:
                    label_list = torch.cat([label_list, torch.zeros((1,1),dtype=torch.float,device=cuda_device)], dim=0)

        label_list = label_list.view(1,-1)[0]
        pred_label = pred_label.view(1,-1)[0]
        print ("label_list",label_list)
        print ("label_list",label_list.shape)

        print("pred_label:",pred_label.shape)
        print("pred_label: ",pred_label)
        if pred_label.shape != label_list.shape:
            print ("Len didn't match")
            p = pred_label.tolist()
            l = label_list.tolist()

        loss = loss_fn(pred_label, label_list)
        print ("loss:", loss)

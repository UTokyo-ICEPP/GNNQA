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
#from tqdm import tqdm

import csv
import time
from sklearn import metrics
from hep_tools import *
import random

torch.manual_seed(0)

import os, sys

import argparse

All_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="choose the model type", type=str)
args = parser.parse_args()

model_name = args.model_name

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

from modules.TrackDataloader import OneTrackDataset,collate_graphs
from modules.dynamic_graph_PhaseIII import Dynamic_Graph_Model

path = '/data/wachan/PhaseIII_data/TrackML_All/'

data_set_train = OneTrackDataset(path, num_start=1000, num_end=1499)
data_set_valid = OneTrackDataset(path, num_start=1500, num_end=1999)

print("data_set_train:",len(data_set_train))
print("data_set_valid:",len(data_set_valid))

train_loader = DataLoader(data_set_train, batch_size=1, shuffle=True, collate_fn=collate_graphs, num_workers=0)
valid_loader = DataLoader(data_set_valid, batch_size=1, shuffle=False, collate_fn=collate_graphs, num_workers=0)

print("train_loader:",len(train_loader),len(train_loader.dataset))
print("valid_loader:",len(valid_loader),len(valid_loader.dataset))

model = Dynamic_Graph_Model(nlayers=3, nclass=1, nodes=24, device=cuda_device)
model_name = 'PhaseIII_test_1sec.pt'

model = model.to(cuda_device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-5)

# ---------------- Make the training loop ----------------- #

train_loss_list, valid_loss_list, test_loss_list = [], [], []
Ind_train_loss_list,Ind_valid_loss_list = [],[]

# number of epochs to train the model
n_epochs = 5

valid_loss_min = np.Inf # track change in validation loss

loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(1, n_epochs+1):
    Epoch_start = time.time()
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    Ind_train_loss = []
    Ind_valid_loss = []
    #train_loss_list = []

    ###################
    # train the model #
    ###################
    #scheduler.step()
    model.train() ## --- set the model to train mode -- ##
    print("=====Training=====")
    if 1:
        for batch, (gr_list, labelpair) in enumerate(train_loader):
            graph_list = []
            sec_loss = 0.0
            for ig in gr_list:
                random.shuffle(ig)

                for i in range(len(ig)):

                    ig[i] = ig[i].to(cuda_device)
                    graph_list.append(ig[i])
                    optimizer.zero_grad()
                    pred_label,edge_array,node_array = model(ig[i])

                    label_list = torch.empty((0),dtype=torch.float,device=cuda_device)
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

                                if ismatch == 1:
                                    break

                            if ismatch == 1:
                                label_list = torch.cat([label_list, torch.ones((1,1),dtype=torch.float,device=cuda_device)], dim=0)
                            else:
                                label_list = torch.cat([label_list, torch.zeros((1,1),dtype=torch.float,device=cuda_device)], dim=0)

                    label_list = label_list.view(1,-1)[0]
                    pred_label = pred_label.view(1,-1)[0]

                    print ("Train, pred_label:",pred_label.shape,pred_label.dtype)
                    print ("Train, label_list:",label_list.shape,label_list.dtype)

                    loss = loss_fn(pred_label, label_list)
                    sec_loss += loss.item()

                    loss.backward(retain_graph=True)

                    del pred_label;
            #=============================================
            optimizer.step()
            scheduler.step(epoch+batch/len(train_loader))

            # update training loss
            train_loss += (sec_loss/len(graph_list))

            del gr_list; del labelpair;
            torch.cuda.empty_cache()

    ######################
    # validate the model #
    ######################
    model.eval()
    print("=====Validation=====")
    if 1:
        nOKc = 0
        nOKf = 0
        nYc = 0
        nYf = 0
        for gr_list, labelpair in valid_loader:
            graph_list = []
            for ig in gr_list:
                for i in range(len(ig)):
                    ig[i] = ig[i].to(cuda_device)
                    graph_list.append(ig[i])

            bg = dgl.batch(graph_list)
            bg = bg.to(cuda_device)

            pred_label,edge_array,node_array = model(bg)

            label_list = torch.empty((0),dtype=torch.float,device=cuda_device)
            for (edge_pair, node_list, labelpair_list_all) in zip(edge_array, node_array, labelpair):

                labelpair_list = []
                for lps in range(len(labelpair_list_all)):
                    for l in labelpair_list_all[lps]:
                        labelpair_list.append(l)

                for idx_ep in range(len(edge_pair[0])):
                    idx_ep0 = edge_pair[0][idx_ep].item()
                    idx_ep1 = edge_pair[1][idx_ep].item()
                    ismatch = 0
                    for lp in labelpair_list:
                        if lp[0] == node_list[idx_ep0].item() and lp[1] == node_list[idx_ep1].item():
                            ismatch = 1

                        if ismatch == 1:
                            break

                    if ismatch == 1:
                        label_list = torch.cat([label_list, torch.ones((1,1),dtype=torch.float,device=cuda_device)], dim=0)
                    else:
                        label_list = torch.cat([label_list, torch.zeros((1,1),dtype=torch.float,device=cuda_device)], dim=0)

            label_list = label_list.view(1,-1)[0]
            pred_label = pred_label.view(1,-1)[0]

            print ("Validation, pred_label:",pred_label.shape,pred_label.dtype)
            print ("Validation, label_list:",label_list.shape,label_list.dtype)
            loss = loss_fn(pred_label, label_list)

            Ind_valid_loss.append(loss.item())
            valid_loss += loss.item()
            #Here we have to first think about how to define TP and FN...
            """
            for (pred,y) in zip(pred_label,label_list):
                print(pred,y)
                act = nn.Sigmoid()
                act_pred = act(pred.item())
                print (act_pred)
                exit()

                if y.item() == 1:
                    nYc += 1
                else:
                    nYf += 1
                if (pred.item() > 0.8):
                    print("1",y.item())
                    if y.item() == 1:
                        nOKc += 1
                else:
                    print("0",y.item())
                    if y.item() == 0:
                        nOKf += 1

            #
            """
            del gr_list; del labelpair; del pred_label;
            torch.cuda.empty_cache()
        #
        #print("Result(valid):",nOKc,nYc,nOKc/nYc,"  ",nOKf,nYf,nOKf/nYf)

    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

    # print training/validation statistics
    #print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        #
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.to('cpu').state_dict(),
            },
            model_name)
        model.to(cuda_device)
        valid_loss_min = valid_loss

    Epoch_end = time.time()
    print ("Epoch", epoch, "finished in", (Epoch_end - Epoch_start), "s")

with open('loss_record_sec1.csv', 'a') as f2:
    writer = csv.writer(f2)
    for i in range(len(train_loss_list)):
        writer.writerow([train_loss_list[i],valid_loss_list[i]])
f2.close()
# ---- end of script ------ #

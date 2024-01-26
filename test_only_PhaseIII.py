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

torch.manual_seed(0)

import os, sys

import argparse

##2024/01/24: Still updating, doesn't work with other PhaseIII script at the moment

All_start = time.time()

#evt_id = 1023

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="choose the model type", type=str)
args = parser.parse_args()
#model_name = args.model_name

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

from modules.TrackDataloader import OneTrackDataset,collate_graphs
from modules.dynamic_graph_2nd import Dynamic_Graph_Model
#path = '/data/wachan/data_10Events/ds100/RZPhi_Regions/'
path = '/data/wachan/data_10Events/ds100/EtaPhi_Small_Test/Eta05/'
#path = '/data/wachan/data_10Events/ds100/EtaPhi_Small_Test/Eta01/'

data_set_tests = OneTrackDataset(path, num_start=7000, num_end=7499)

print("data_set_tests:",len(data_set_tests))

tests_loader = DataLoader(data_set_tests, batch_size=1, shuffle=False, collate_fn=collate_graphs, num_workers=2)

print("tests_loader:",len(tests_loader),len(tests_loader.dataset))

#model = Dynamic_Graph_Model(nlayers=3, nclass=2, nodes=16, device=cuda_device)
model = Dynamic_Graph_Model(nlayers=4, nclass=2, nodes=24, device=cuda_device)
model_name = '202309230_25Event_Eta015_HPO_20Epochs_Trial2.pt'
out_name = '20231005_25Event_Eta05_HPO_20Epochs_Trial1_PredictionModel.h5'

model = model.to(cuda_device)

loss_fn = nn.CrossEntropyLoss()

Nom = nn.Softmax(dim=0)

if 1:
    # test
    y_true = np.empty(0)
    y_pred = np.empty(0)
    y_pred_0 = np.empty(0)
    y_pred_mix = np.empty(0)
    y_pred_binary = np.empty(0)
    y_tp = np.empty(0)
    y_fp = np.empty(0)
    y_tn = np.empty(0)
    y_fn = np.empty(0)
    edge_arr = np.empty(0)
    test_loss = 0.0
    print("model.training=",model.training)
    if 1:
        # load the best parameter set
        checkpoint = torch.load(model_name, map_location=torch.device(cuda_device))
        model.load_state_dict(checkpoint["model_state_dict"])
        #print (checkpoint)
        best_epoch = checkpoint["epoch"]
        print("Use a model of epoch", best_epoch, "for test")
        #exit()
        #
        nOKc = 0
        nOKf = 0
        nYc = 0
        nYf = 0
        nWc = 0
        nWf = 0

        for gr_list, labelpair in tests_loader:
            graph_list = []
            for ig in gr_list:
                for i in range(len(ig)):
                    ig[i] = ig[i].to(cuda_device)
                    graph_list.append(ig[i])

            bg = dgl.batch(graph_list)
            bg = bg.to(cuda_device)
            #print("Test bg",bg)

            pred_label,edge_array,node_array = model(bg)
            #print (edge_array.dtype)
            print ("edge_array: ",edge_array)
            #print (edge_array[0][0]) #src
            #print (edge_array[0][1]) #std

            src_list = edge_array[0][0].tolist()
            std_list = edge_array[0][1].tolist()
            print ("src_list: ",src_list) #src
            print ("std_list: ",std_list) #std

            print ("node_array: ",node_array)
            #print (node_array[0][edge_array[0][0][0].item()],node_array[0][edge_array[0][1][0].item()])#This should directly give us hit pairs

            edge_list = []
            for i in range(len(src_list)):
            #for i in range(0,10):
                #hit_pair = []
                h0 = int(node_array[0][src_list[i]].item())
                h1 = int(node_array[0][std_list[i]].item()) #Edge num, src, std
                hit_pair = [h0,h1]
                print ("edge number: ",i," hit_pair: ", hit_pair)
                edge_list.append(hit_pair)

            #print (edge_list)
            #edge_tensor = torch.tensor(edge_list)
            #exit()
            #for i in range(len(src_list)):


            label_list = torch.empty((0),dtype=torch.long,device=cuda_device)
            #print (label_list.shape)
            for lps in range(len(labelpair)):
                #print (labelpair[lps])
                for i_lp in labelpair[lps]:
                    #print (i_lp)
                    i_lp = i_lp.to(cuda_device)
                    #print (i_lp.shape)
                    label_list = torch.cat((label_list,i_lp), dim=0)

            #pred_label = pred_label.view(1,-1)[0]
            loss = loss_fn(pred_label, label_list)
            test_loss += loss.item()
            #

            for (pred,y) in zip(pred_label,label_list):
                #pred = Nom(pred_raw)
                #edge_arr = np.append(edge_arr,edges.to('cpu').detach().numpy().copy())
                #edge_arr = np.append(y_true,y.to('cpu').detach().numpy().copy())
                y_true = np.append(y_true,y.to('cpu').detach().numpy().copy())
                y_pred = np.append(y_pred,pred[1].to('cpu').detach().numpy().copy())
                y_pred_0 = np.append(y_pred_0,pred[0].to('cpu').detach().numpy().copy())
                y_pred_binary = np.append(y_pred_binary,pred.argmax(0).to('cpu').detach().numpy().copy())
                #print ("y_true: ",y_true)
                #print ("y_pred: ",y_pred)
                #print ("y_pred_binary: ",y_pred_binary)
                #exit()
                #print("pred,y:",pred[1].item(),y.item())
                #print("pred,y(max):",pred[1].item(),y.item(),pred.argmax(0))
                #
                #print(pred,y)


                if y.item() == 1:
                    nYc += 1
                else:
                    nYf += 1

                if (pred[0].item() < pred[1].item()):

                    y_pred_mix = np.append(y_pred_mix,pred[1].to('cpu').detach().numpy().copy())
                    print ("1",pred[0].item(),pred[1].item())
                #if pred.item() < 0.5:
                    #print("1",y.item())
                    if y.item() == 1:#TP, pred = 1 and true = 1
                        nOKc += 1
                        y_tp = np.append(y_tp,pred[1].to('cpu').detach().numpy().copy())
                    elif y.item() == 0:#FP, pred = 1 and true = 0
                        nWc += 1
                        y_fp = np.append(y_fp,pred[1].to('cpu').detach().numpy().copy())
                else:
                    y_pred_mix = np.append(y_pred_mix,pred[0].to('cpu').detach().numpy().copy())
                    print ("0",pred[0].item(),pred[1].item())
                    #print("0",y.item())
                    if y.item() == 0:#TN, pred = 0 and true = 0
                        nOKf += 1
                        y_tn = np.append(y_tn,pred[1].to('cpu').detach().numpy().copy())
                    elif y.item() == 1:#FN, pred = 0 and true = 1
                        nWf += 1
                        y_fn = np.append(y_fn,pred[1].to('cpu').detach().numpy().copy())

                ####Pick up edge Num...?


                print (pred[0].item(),pred[1].item(),pred.argmax(0),y.item())
                print ("nYc: ", nYc, "TP: ", nOKc, "FP: ", nWc)
                print ("nYf: ", nYf, "TN: ", nOKf, "FN: ", nWf)
                if (nOKc+nWf) != 0 and (nOKf+nWc) != 0:
                    print ("TPR: ", nOKc/(nOKc+nWf), "FPR: ", nWc/(nOKf+nWc))

            del gr_list; del labelpair; del pred_label;
            torch.cuda.empty_cache()
        #
        print("Result(test):",nOKc,nYc,nOKc/nYc,"  ",nOKf,nYf,nOKf/nYf)
        print(len(y_true),len(y_pred),len(y_pred_binary))
        print("y_true",y_true)
        print("y_pred",y_pred)

        #
        test_loss = test_loss/len(tests_loader.dataset)
        print('Test Loss: {:.6f}'.format(test_loss))
        #
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, drop_intermediate=False)

        roc_auc = metrics.auc(fpr, tpr)
        accuracy = metrics.accuracy_score(y_true,y_pred_binary)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_binary).ravel()
        #for i in range(len(tpr)):
            #print (i, "tpr, fpr, thresholds:", tpr[i], fpr[i], thresholds[i])
        print("AUC:",roc_auc)
        print("ACC:",accuracy)
        # Correct answer: TP,TN
        # Wrong answer: FP,FN
        print("TP(true=1),TN(true=0),FP(true=0),FN(true=1):",tp,tn,fp,fn)
        #make_ROC_HEP(roc_auc,tpr,fpr,thresholds,"./Output/Test_Only_Variation_Trial/25Event_HPO_20Epochs_Model/Event"+str(evt_id)+"_Only_Eta015_Output/ROC.png","./Output/Test_Only_Variation_Trial/25Event_HPO_20Epochs_Model/Event"+str(evt_id)+"_Only_Eta015_Output/RvsEff.png")
        make_ROC_HEP(roc_auc,tpr,fpr,thresholds,"./Output/Test_Only_Variation_Trial/25Event_HPO_20Epochs_Model/Event"+str(evt_id)+"_Only_Eta05_Output/ROC_W0_Edges_Eta02.png","./Output/Test_Only_Variation_Trial/25Event_HPO_20Epochs_Model/Event"+str(evt_id)+"_Only_Eta05_Output/RvsEff_W0_Edges_Eta02.png")

        #hf = h5py.File("./Output/Test_Only_Variation_Trial/25Event_HPO_20Epochs_Model/Event"+str(evt_id)+"_Only_Eta015_Output/"+str(out_name), 'w')
        hf = h5py.File("./Output/Test_Only_Variation_Trial/25Event_HPO_20Epochs_Model/Event"+str(evt_id)+"_Only_Eta05_Output/"+str(out_name), 'w')

        hf.create_dataset('y_true', data=y_true)
        hf.create_dataset('y_pred_mix', data=y_pred_mix)
        hf.create_dataset('y_pred', data=y_pred)
        hf.create_dataset('y_pred_0', data=y_pred_0)
        hf.create_dataset('TP', data=y_tp)
        hf.create_dataset('FP', data=y_fp)
        hf.create_dataset('TN', data=y_tn)
        hf.create_dataset('FN', data=y_fn)
        hf.create_dataset('Edges', data=edge_list)
        hf.create_dataset('Edges_src', data=src_list)
        hf.create_dataset('Edges_std', data=std_list)
        hf.close()

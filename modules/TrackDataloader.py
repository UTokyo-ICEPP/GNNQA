import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch

import dgl
from dgl import backend as F
#from dgl import RemoveSelfLoop

import math
import time
import csv
from torch.utils.data import Dataset, DataLoader, Sampler
from dgl.data.utils import save_graphs, load_graphs
from pathlib import Path


def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def make_graph_structure(num_nodes,VolLayID,hit_Z,hit_r,hit_phi,hit_eta):
    nodes_src = []
    nodes_dst = []
    #print (num_nodes,VolLayID,hit_Z,hit_r,hit_phi,hit_eta)
    for i in range(num_nodes-1):
        for j in range(i+1,num_nodes):
            if VolLayID[j] == VolLayID[i] +1:
                dr = hit_r[j] - hit_r[i]
                dz = hit_Z[j] - hit_Z[i]
                dPhi = calc_dphi(hit_phi[i],hit_phi[j])
                dEta = hit_eta[j] - hit_eta[i]
                #phi_slope = dPhi / dr
                #z0 = hit_Z[i] - hit_r[i] * dz / dr
                if abs(dPhi) <= 0.2:
                    if abs(dEta) <= 0.2:

                        if VolLayID[j] <= 3:
                            if abs(dz) <= 100:
                                if abs(dr) <= 100:
                                    nodes_src.append(i)
                                    nodes_dst.append(j)

                        elif 3 < VolLayID[j] <= 7:
                            if abs(dz) <= 200:
                                if abs(dr) <= 200:
                                    nodes_src.append(i)
                                    nodes_dst.append(j)

                        elif VolLayID[j] > 7:
                            if abs(dz) <= 200:
                                if abs(dr) <= 250:
                                    nodes_src.append(i)
                                    nodes_dst.append(j)



    return nodes_src, nodes_dst

def Calcul_Edge(gr,NumEdge,hit_z,hit_r,hit_phi,hit_eta):
    #If we consider edge features, this is the parameters used in the original QUBO pre-selection
    dz_list = []
    dr_list = []
    dPhi_list = []
    dEta_list = []
    for i in range(NumEdge):
        #print (i)
        #print (gr.find_edges(i))
        Src = gr.find_edges(i)[0].item()
        Dst = gr.find_edges(i)[1].item()

        #From here call the src and dts node number and use it to calcualte the rest

        dr = hit_r[Dst] - hit_r[Src]
        dz = hit_z[Dst] - hit_z[Src]
        dPhi = calc_dphi(hit_phi[Src],hit_phi[Dst])
        dEta = hit_eta[Dst] - hit_eta[Src]

        dz_list.append(dz)
        dr_list.append(dr)
        dPhi_list.append(dPhi)
        dEta_list.append(dEta)

    return dr_list,dz_list,dPhi_list,dEta_list

class OneTrackDataset(Dataset):
    def __init__(self, filepath, num_start, num_end):

        self.path = filepath
        self.num_start = num_start
        self.num_end = num_end

        self.n_eff = self.num_end - self.num_start + 1

    def __len__(self):
        return self.n_eff

    def __getitem__(self, event_idx):
        print ("====Dataloader====")

        event_idx = self.num_start + event_idx
        print('Event_idx',event_idx)

        file_name = Path('/data/wachan/PhaseIII_data/TrackML_All/event00000'+str(event_idx)+'-hits.csv')

        print ('Trying to read: ',file_name)

        if file_name.is_file():

            Saving_path = '/data/wachan/PhaseIII_network/Saved_Graph_Label/Single_section/'

            label_file = Path(Saving_path+'event00000'+str(event_idx)+'-Label.csv')
            graph_file = Path(Saving_path+'event00000'+str(event_idx)+'-Graph.bin')

            load_start = time.time()
            label_list = []
            doublets = pd.read_csv(self.path + 'event00000'+str(event_idx)+'-pT08_matched-doublets.csv')

            Doublet_Pair = torch.from_numpy(doublets.to_numpy()[:])
            label_list.append(Doublet_Pair)

            if graph_file.is_file():
                print ('graph file exist')

                gr = load_graphs(Saving_path+'event00000'+str(event_idx)+'-Graph.bin')
                gr_list = gr[0]

            else:
                gr_list = []
                print ('No saved files, starting to generate graph and label...')

                hits= pd.read_csv(self.path + 'event00000'+str(event_idx)+'-sampling-pT08_matched-hits.csv')

                hit_r = torch.tensor(hits.to_numpy()[:,6])
                hit_phi = torch.tensor(hits.to_numpy()[:,7])
                hit_z = torch.tensor(hits.to_numpy()[:,2])

                pos = torch.stack([hit_r, hit_phi, hit_z], dim=-1).float()
                nhit = len(pos)

                # node features

                hitID = torch.LongTensor(hits.to_numpy()[:,1])
                hit_eta = torch.tensor(hits.to_numpy()[:,8])
                VolLayID = torch.LongTensor(hits.to_numpy()[:,3])


                feat = torch.stack(
                    [
                        hit_eta,
                        VolLayID
                    ]
                    , dim=-1).float()

                g = dgl.graph(make_graph_structure(len(hitID),VolLayID,hit_z,hit_r,hit_phi,hit_eta),num_nodes = len(hitID))

                gr = dgl.to_bidirected(g)

                NumEdge = gr.number_of_edges()

                gr.ndata['idx'] = hitID
                gr.ndata['x'] = pos
                gr.ndata['en'] = feat

                print (gr)
                gr_list.append(gr)

                print ("Saving graph...")
                print (gr)

                save_graphs(Saving_path+'event00000'+str(event_idx)+'-Graph.bin',gr)

                label_save = label_list
                print (label_save)

                print ("Saving label list...")
                with open(Saving_path+'event00000'+str(event_idx)+'-Label.csv','w') as f:
                    writer = csv.writer(f)
                    writer.writerow(label_save) #Savea as 0,0,0,0,0,0,...,0
                f.close()


            load_end = time.time()
            print ("Graph generated in ", (load_end - load_start), "s")
            print ("gr_list: ",gr_list)

            return { 'gr' : gr_list,
                     'y' : label_list }

        else:
            print ('Event missing')


def collate_graphs(event_list) :
    #print ("=======Collate_graphs=======")
    graph = []
    labelpair = []
    for item in event_list:
        if item is not None:
        #Skipping the event ID which have no sample
            graph.append(item['gr'])
            labelpair.append(item['y'])

    return graph,labelpair

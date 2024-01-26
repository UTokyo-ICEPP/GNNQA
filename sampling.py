import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import csv
import pandas as pd
import math
import torch
import dgl
import pickle
import itertools
import matplotlib.pylab as pl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_deta(hitpair):
    r1 = hitpair.r_1
    r2 = hitpair.r_2
    z1 = hitpair.z_1
    z2 = hitpair.z_2

    R1 = np.sqrt(r1**2 + z1**2)
    R2 = np.sqrt(r2**2 + z2**2)
    theta1 = np.arccos(z1/R1)
    theta2 = np.arccos(z2/R2)
#     theta1 = np.arctan(r1/z1)
#     theta2 = np.arctan(r2/z2)
    eta1 = -np.log(np.tan(theta1/2.0))
    eta2 = -np.log(np.tan(theta2/2.0))
    return eta1 - eta2

def output_hits(hits):

    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    theta = np.arctan2(r,hits.z)
    eta = -1*(np.log(np.tan(theta/2)))
    # Select the data columns we need
    hits = (hits[['hit_id', 'z', 'layer']]
            .assign(r=r, phi=phi,eta=eta,theta=theta))
    return hits

def select_truth(hits):
    #hit_id	particle_id	tx	ty	tz	tpx	tpy	tpz	weight
    # Barrel volume and layer ids

    r = np.sqrt(hits.tx**2 + hits.ty**2)
    pT = np.sqrt(hits.tpx**2 + hits.tpy**2)
    phi = np.arctan2(hits.ty, hits.tx)
    theta = np.arctan2(r,hits.tz)
    eta = -1*(np.log(np.tan(theta/2)))
    # Select the data columns we need
    hits = (hits[['hit_id','particle_id','tx', 'ty','tz','tpx','tpy','tpz','weight']]
            .assign(r=r, phi=phi,eta=eta,theta=theta,pT=pT))
    return hits

def select_hits(hits):
    # Barrel volume and layer ids
    vlids = [(8,2), (8,4), (8,6), (8,8),
             (13,2), (13,4), (13,6), (13,8),
             (17,2), (17,4)]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])

    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    theta = np.arctan2(r,hits.z)
    eta = -1*(np.log(np.tan(theta/2)))
    # Select the data columns we need
    hits = (hits[['hit_id', 'z', 'layer','x','y']]
            .assign(r=r, phi=phi,eta=eta,theta=theta))
    return hits

def get_doublets(hits, gid_keys,gid_start, gid_end):

    # Group hits by geometry ID
    hit_gid_groups = hits.groupby(gid_keys)

    doublets = []

    # Loop over geometry ID pairs
    for gid1, gid2 in zip(gid_start, gid_end):
        hits1 = hit_gid_groups.get_group(gid1)
        hits2 = hit_gid_groups.get_group(gid2)
        #print (gid1,gid2)
        #print (hits1,hits2)

        hit_pairs = pd.merge(hits1.reset_index(), hits2.reset_index(),on='evtid', suffixes=('_1', '_2'))

        # Calculate coordinate differences
        dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
        deta = calc_deta(hit_pairs)
        dz = hit_pairs.z_2 - hit_pairs.z_1
        dr = hit_pairs.r_2 - hit_pairs.r_1
        phi_slope = dphi / dr
        z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
        rz = np.arctan2(dz, dr)
        dLay = hit_pairs.layer_2 - hit_pairs.layer_1

        # Identify the true pairs
        y = (hit_pairs.label_1 == hit_pairs.label_2) & (hit_pairs.label_1 != 0)

        # Put the results in a new dataframe
        doublets.append(hit_pairs[['evtid', 'index_1', 'index_2', 'layer_1', 'layer_2']]
                        .assign(dphi=dphi, dz=dz, dr=dr, deta=deta, y=y, dLayer = dLay, phi_slope=phi_slope, z0=z0,rz=rz,z_1=hit_pairs.z_1,z_2=hit_pairs.z_2,r_1=hit_pairs.r_1,r_2=hit_pairs.r_2))

    return pd.concat(doublets, ignore_index=True)

#======================================================================
for ds in range(2400,10000):
    print ("====event "+str(ds)+"====")
    file_name = Path('/data/wachan/PhaseIII_data/TrackML_All/event00000'+str(ds)+'-hits.csv')

    if file_name.is_file():
        hits =  pd.read_csv('/data/wachan/PhaseIII_data/TrackML_All/event00000'+str(ds)+'-hits.csv')
        truth =  pd.read_csv('/data/wachan/PhaseIII_data/TrackML_All/event00000'+str(ds)+'-truth.csv')
        particles =  pd.read_csv('/data/wachan/PhaseIII_data/TrackML_All/event00000'+str(ds)+'-particles.csv')

        print ("#Hits (Raw): ",len(hits))
        print ("#Truth Hits (Raw): ",len(truth))
        print ("#Particles (Raw): ",len(particles))

        #Target extraction
        #Cond 1: particles with 4 hits
        Track_particles = particles[particles.nhits >= 4]
        particle_list = particles["particle_id"].values.tolist()

        #truth hits: add parameters
        truth_hits = (select_truth(truth).reset_index(drop=True))

        #Cond 2: pT > 0.8 GeV
        truth_hits_pT = truth_hits[truth_hits.pT > 0.8]

        #Cond 3: |eta| <= 1
        truth_hits_Eta = truth_hits_pT[abs(truth_hits_pT.eta) <= 1]

        #matching: Only keep particles fullfillinf both Cond 1 and 2
        truth_track_list = []
        for i in range(len(particle_list)):

            truth_track_hits = truth_hits_Eta[truth_hits_Eta.particle_id == particle_list[i]]
            #print (i, truth_track_hits)
            if  not truth_track_hits.empty:
                i_truth_track = truth_track_hits["hit_id"].values.tolist()
                if len(i_truth_track) >= 4:
                    #print (i_truth_track)
                    truth_track_list.append(i_truth_track)

        print ("#Tracks with #hits > 4: ",len(truth_track_list))
        #print (truth_track_list)

        #Decompose the list and form the target doublet list
        track_doublet = [] #This is our target list...
        for i in range(len(truth_track_list)):
            for j in range(len(truth_track_list[i])-1):
                j_doublet = []
                #print (i,j, j+1, truth_track_list[i], truth_track_list[i][j], truth_track_list[i][j+1])
                j_doublet.append(truth_track_list[i][j])
                j_doublet.append(truth_track_list[i][j+1])
                #print (j_doublet)
                track_doublet.append(j_doublet)

        print ("#Matched doublets: ",len(track_doublet))

        headers_T = ['strat','end']

        OutName_T = '/data/wachan/PhaseIII_data/TrackML_All/event00000'+str(ds)+'-pT08_matched-doublets.csv'

        with open(OutName_T,'w') as f3:
            writer = csv.writer(f3)
            writer.writerow(headers_T)
            writer.writerows(track_doublet)
        f3.close()

        truth_hits_soft_pT = truth_hits[truth_hits.pT > 0]
        truth_hits_soft_pT_index = truth_hits_soft_pT.reset_index(drop=True)

        hits = (select_hits(hits).reset_index(drop=True))
        label = 0
        hits = (hits.assign(label = label))

        Target_Hit_list = []
        for i in range(len(truth_hits_soft_pT_index)):
            Target_Hit_list.append(truth_hits_soft_pT_index['hit_id'][i])

        #print (len(Target_Hit_list))

        for j in range(len(Target_Hit_list)):
            Target = int(Target_Hit_list[j])
            hits.loc[hits['hit_id'] == Target, 'label'] = 1

        matched_hits = hits[hits.label == 1]
        #print (len(matched_hits))

        eta_hits = matched_hits[abs(matched_hits.eta) <= 1]
        #print (len(eta_hits))

        matched_hits_out = eta_hits.reset_index(drop=True)
        #matched_hits_out = matched_hits_out.iloc[: , 1:]

        matched_hits_out.to_csv('/data/wachan/PhaseIII_data/TrackML_All/event00000'+str(ds)+'-pT08_matched-hits.csv')

        Full_hits = pd.read_csv('/data/wachan/PhaseIII_data/TrackML_All/event00000'+str(ds)+'-pT08_matched-hits.csv')
        #Split them into regions: [-pi, -1.9], [-2, -0.9], [-1, 0.1], [0,1.1], [1,2.1], [2, pi]
        Full_hits = Full_hits.reset_index(drop=True)
        Full_hits = Full_hits.iloc[: , 1:]
        Region1_hits = Full_hits[(Full_hits.phi >= 0) & (Full_hits.phi < 1) & (Full_hits.eta >= 0) & (Full_hits.eta < 0.5)].reset_index(drop=True)

        print ("#Sampled hits: ",len(Region1_hits))

        Region1_hits.to_csv('/data/wachan/PhaseIII_data/TrackML_All/event00000'+str(ds)+'-sampling-pT08_matched-hits.csv')

    else:
        print ('Event missing')

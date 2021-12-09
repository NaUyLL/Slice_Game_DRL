"""
author:Luo Liyuan
"""
import torch
import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import cdist

def compute_badw_ch(ue_info, bs_info):
    # -------------RB initial calculation------------
    Pt = 30  # TP 's transmitting power in dBm
    TP_subchannel = 5
    TP_freq = 2400  # TP frequencedddd
    TP_Bndw = 5000000  # bandwidth in Hz
    Nsd = -174 # noise spectral density in dBm/Hz
    # -------------resource block data------------
    subch_Pt = Pt / TP_subchannel
    RB_Bndw = TP_Bndw / TP_subchannel
    Nsd_mW = math.pow(10, (Nsd / 10))  # --in mW--convert in =
    Nsd_subch = Nsd_mW * (TP_Bndw / TP_subchannel)
    NoB_subch = 10 * math.log10(Nsd_subch)

    ue_x = ue_info[0]
    ue_y = ue_info[1]
    rate_demand = ue_info[2]

    bs_x = bs_info[0]
    bs_y = bs_info[1]

    distance = cdist(np.array([[ue_x, ue_y]]), np.array([[bs_x, bs_y]]),
                     metric='euclidean')
    # 计算路径损耗
    path_loss = (20 * math.log(distance[0, 0])) + (20 * math.log(TP_freq)) - 27.55
    rij = subch_Pt - path_loss# - np.random.randint(1, 10)  # rssi nith nic of ith UE to jth AP
    print(rij)
    RB_Thr = RB_Bndw  * math.log2(1 + (rij / NoB_subch))  # bps
    Need_badw = rate_demand * 1000 / RB_Thr# * RB_Bndw
    return Need_badw

def compute_badw(ue_info, bs_info):
    # -------------RB initial calculation------------
    Pt = 30  # TP 's transmitting power in dBm
    TP_freq = 2400  # TP frequencedddd
    TP_Bndw = 5000000 #5000000  # bandwidth in Hz
    Nsd = -174  # noise spectral density in dBm/Hz
    Nsd_mW = math.pow(10, (Nsd / 10))  # --in mW--convert in =
    NoB_EM = 10 * math.log10(Nsd_mW)
    ue_x = ue_info[0]
    ue_y = ue_info[1]
    rate_demand = ue_info[2]

    bs_x = bs_info[0]
    bs_y = bs_info[1]

    distance = cdist(np.array([[ue_x, ue_y]]), np.array([[bs_x, bs_y]]),
                     metric='euclidean')

    path_loss = (20 * math.log(distance[0, 0])) + (20 * math.log(TP_freq)) - 27.55
    rij = Pt/TP_Bndw - path_loss# - np.random.randint(1, 10)
    print(rij)
    EM_Thr = math.log2(1 + (rij / NoB_EM))  # bps

    Need_badw = rate_demand * 1000 / EM_Thr
    return Need_badw

def generate_ue(ue_num):
    columns_ue = ['ue_x', 'ue_y', 'rate_demand', 'ue_id']
    ue_df = pd.DataFrame(index=np.arange(ue_num),
                         columns=columns_ue)
    ue_id = 0
    for ui in range(ue_num):
        deta = np.random.randint(0, 360)
        r = np.random.randint(0, 80)
        ue_X = 350 + r * math.cos(math.radians(deta))
        ue_Y = 350 + r * math.sin(math.radians(deta))
        ue_df.loc[ue_id] = [ue_X, ue_Y, 500, ue_id] #rate_demands
        ue_id += 1

    return ue_df

def generate_bs(bs_num):
    teta = 0
    r = 120
    columns_bs = ['x', 'y', "bs_id"]#, 'backhaul']
    TPs = pd.DataFrame(index=np.arange(bs_num), columns=columns_bs)
    bs_id = 0
    for bi in range(bs_num):
        x = 350 + r * math.cos(math.radians(teta))
        y = 350 + r * math.sin(math.radians(teta))
        teta += 90
        #backhaul = np.random.randint(30, 40) * 1e6  # backhaul data rate /bps
        TPs.loc[bi] = [x, y, bs_id]#, backhaul]
        bs_id += 1
    return TPs

if __name__ == "__main__":
    TPs = generate_bs(2)
    UEs = generate_ue(1)

    print(TPs)
    print(UEs)
    bs1 = np.array(TPs.loc[0, ['x', 'y']])
    bs2 = np.array(TPs.loc[1, ['x', 'y']])
    ue1 = np.array(UEs.loc[0, ['ue_x', 'ue_y', 'rate_demand']])
    print(bs1)
    print(bs2)
    print(ue1)
    print(compute_badw_ch(ue1, bs1))
    print(compute_badw(ue1, bs1))

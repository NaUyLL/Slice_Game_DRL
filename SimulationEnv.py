"""
author:Luo Liyuan
"""
import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import cdist

def compute_badw(ue_info, bs_info):
    # -------------RB initial calculation------------
    Pt = 30  # TP 's transmitting power in dBm
    TP_freq = 2400  # TP frequencedddd
    TP_Bndw = 5 #5000000  # bandwidth in Hz
    Nsd = -174  # noise spectral density in dBm/Hz
    EM = 1 # 1MHz
    EM_Pt = Pt / TP_Bndw
    Nsd_mW = math.pow(10, (Nsd / 10))  # --in mW--convert in =
    Nsd_EM = Nsd_mW * (TP_Bndw / EM)
    NoB_EM = 10 * math.log10(Nsd_EM)
    ue_x = ue_info[0]
    ue_y = ue_info[1]
    rate_demand = ue_info[2]

    bs_x = bs_info[0]
    bs_y = bs_info[1]

    distance = cdist(np.array([[ue_x, ue_y]]), np.array([bs_x, bs_y]).reshape(1, 2),
                     metric='euclidean')

    path_loss = (20 * math.log(distance[0, 0])) + (20 * math.log(TP_freq)) - 27.55
    rij = EM_Pt - path_loss - np.random.randint(1, 10)
    EM_Thr = EM * math.log2(1 + (rij / NoB_EM))  # bps

    Need_badw = rate_demand * 1000 / EM_Thr
    return Need_badw


class SimulationEnv(object):
    bs_num = 2  # bs数量
    slice_num = 2  # 切片数量
    sys_bandwidth = 5000000  # 系统带宽 HZ
    rate_demands = 100  # / kbps [10~25] 用户rate_demand可以不同

    # 产生UE信息
    def generate_ue(self, ue_num):
        columns_ue = ['ue_x', 'ue_y', 'rate_demand', 'ue_id']
        ue_df = pd.DataFrame(index=np.arange(sum(ue_num)),
                             columns=columns_ue)
        ue_id = 0
        for ui in range(ue_num):
            deta = np.random.randint(0, 360)
            r = np.random.randint(0, 80)
            ue_X = 350 + r * math.cos(math.radians(deta))
            ue_Y = 350 + r * math.sin(math.radians(deta))
            ue_df.loc[ue_id] = [ue_X, ue_Y, self.rate_demands, ue_id]
            ue_id += 1

        return ue_df

    # 产生BS信息
    def generate_bs(self, bs_num):
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
    pass
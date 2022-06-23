import sys
sys.path.append("../")

from utils import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import allib

QTDE_EXP = 10
TRAIN_SIZE = 30
SEED = 13
LBL_QTDE = [30]

out = get_emb_vec("clipsum")

bacc_list = np.zeros([QTDE_EXP], dtype=np.float32)
f1_list = np.zeros([QTDE_EXP], dtype=np.float32)

for i in range(QTDE_EXP):
    exp = allib.kmeans_sampling(1, LBL_QTDE, i)

    bacc_list[i], _, f1_list[i], _ = exp.sample_ssl(img_emb, img_anno, 
                                                    itera=None)

bacc_mean = "BAcc - Mean: {}".format(np.mean(bacc_list, axis=0))
bacc_std = "BAcc - Std: {}".format(np.std(bacc_list, axis=0))
f1_mean = "F1 - Mean: {}".format(np.mean(f1_list, axis=0))
f1_std = "F1 - Std: {}".format(np.std(f1_list, axis=0))

print(bacc_mean)
print(bacc_std)
print(f1_mean)
print(f1_std)



    


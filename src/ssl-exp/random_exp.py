import sys
sys.path.append("../")

from utils import *
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import allib

QTDE_EXP = 10
TRAIN_SIZE = 30
SEED = 13
LBL_QTDE = [30]

out = get_emb_vec()

img_emb = out['img_emb']
img_anno = out['img_anno']

exp = allib.random_sampling(QTDE_EXP, LBL_QTDE, SEED)

exp.sample_ssl(img_emb, img_anno, itera=None, display=False)


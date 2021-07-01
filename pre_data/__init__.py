import os
import copy
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import scipy.io as sio
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

#PATH_ori = 'E:/program/mine/data/ori_data/'
PATH_ori = os.getcwd()+'/data/ori_data/'
#PATH_pre = 'E:/program/mine/data/pre_data/'
PATH_pre = os.getcwd()+'/data/pre_data/'
#PATH_graph = 'E:/program/mine/data/graph_data/'
PATH_graph = os.getcwd()+'/data/graph_data/'

CNN_stride=1
Num_epoch=5
Learn_rate=0.002

NUM_adj_sample_spa_window=7  #必须是奇数
NUM_adj_sample_spe_threshold=0.95
NUM_adj_sample_spe_retain=7
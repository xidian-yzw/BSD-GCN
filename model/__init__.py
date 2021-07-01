import numpy as np
import tensorflow as tf

LAYER_UIDS = {}  # 用于图层名称分配的全局唯一图层ID字典
Allowed_kwargs = {'Basic_gcn_layer', 'Bs_mask_layer', 'Bs_layer', 'Last_layer', 'Fc_layer'}
NUM_dropout=0.5
NUM_select_bs=60
NUM_block=5
LAYER_ouput_dim=256
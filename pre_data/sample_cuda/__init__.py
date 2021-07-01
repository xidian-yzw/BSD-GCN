import numpy as np
import torch

from torch.autograd import Function
import sys
import  os
import GCN_Sample_split
# import GCN_Sample_split

# class SampleFunction(Function):
#
#     @staticmethod
#     def forward(ctx,features,features_loc,spa_window,spe_threshold,spe_retain):
#         # features = torch.from_numpy(features.astype(np.float16)).cuda()
#         # features_loc = torch.from_numpy(features_loc.astype(np.float16)).cuda()
#         spa_spe = torch.Tensor([spa_window,spe_threshold],type=torch.float16).cuda()
#
#         output = GCN_Sample.forward(features, features_loc,spa_spe)
#
#         adj_sample_spa = output[0]
#         adj_sample_spe = output[1]
#
#         b = adj_sample_spe.detach()
#         b = torch.sort(b,dim=1).values
#         adj_sample_spe = torch.where(adj_sample_spe>b[:,-(spe_retain+1)],adj_sample_spe,torch.zeros_like(adj_sample_spe))
#         return adj_sample_spa,adj_sample_spe.t()

class SampleSplitFunction(Function):

    @staticmethod
    def forward(ctx,features,features_loc, spa_spe, select):
        spa, spe = GCN_Sample_split.forward(features, features_loc, spa_spe, select)
        return spa.to(torch.float16), spe.to(torch.float16)

def cuSample(features,features_loc,spa_window,spe_threshold,spe_retain):
    features = torch.from_numpy(features).float().cuda()
    num_sample = features.size()[0]
    features_loc = torch.from_numpy(features_loc).float().cuda()
    spa_spe = torch.Tensor([spa_window, spe_threshold]).float().cuda()
    index = torch.arange(num_sample, dtype=torch.float32).cuda()

    split_num = int(np.ceil((num_sample ** 2) / 2e8))
    split_len = (num_sample) // split_num

    adj_sample_spa = np.zeros([num_sample, num_sample],dtype=np.float16)
    adj_sample_spe = np.zeros([num_sample, num_sample],dtype=np.float16)
    for ind in range(split_num+1):
        select = index[ind * split_len:] if (ind + 1) * split_len > num_sample else index[ind * split_len:(ind + 1) * split_len]
        if select.size()[0]==0:
            continue
        spa,spe=SampleSplitFunction.apply(features,features_loc,spa_spe,select)
        select = select.int().cpu().numpy()
        adj_sample_spa[select] = spa.cpu().numpy()
        adj_sample_spe[select] = spe.cpu().numpy()
    adj_sample_spa = adj_sample_spa+adj_sample_spa.T
    adj_sample_spe = adj_sample_spe+adj_sample_spe.T
    eye = np.eye(num_sample,dtype=bool)
    adj_sample_spa[eye]/=2
    adj_sample_spe[eye]/=2
    b=np.sort(adj_sample_spe,axis=1)
    adj_sample_spe = np.where(adj_sample_spe> b[:,-(spe_retain + 1)],adj_sample_spe, 0)
    return adj_sample_spa,adj_sample_spe.T

# def cuSample(features,features_loc,spa_window,spe_threshold,spe_retain):
#     adj_sample_spa,adj_sample_spe=SampleFunction.apply(features,features_loc,spa_window,spe_threshold,spe_retain)
#     return adj_sample_spa.cpu().numpy(),adj_sample_spe.cpu().numpy()
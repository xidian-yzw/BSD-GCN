import os
import copy
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from pre_data.get_prefile import Get_pre_file
from pre_data.pre_to_graph import Get_graph_file

def pre_data(data_set,ratio):
   # Get_graph_file(data_set)
   Get_pre_file(data_set, ratio)

def load_data(data_name):

    # read data
    pre = sio.loadmat(os.getcwd() + '/data/graph_data/' + data_name + '_graph.mat')

    features = pre['features']
    adj_sample_spe = pre['adj_sample_spe']
    adj_sample_spa = pre['adj_sample_spa']
    adj_bs_spe = pre['adj_bs_spe']
    adj_bs_spa = pre['adj_bs_spa']
    
    pre = sio.loadmat(os.getcwd() + '/data/pre_data/' + data_name + '_pre.mat')
    
    y_train = pre['y_train']
    y_test = pre['y_test']
    train_mask = pre['train_mask']
    test_mask = pre['test_mask']
    label = pre['label']

    # data processing
    features = sp.csr_matrix(features)
#    adj_sample_spe = sp.csr_matrix(adj_sample_spe)
#    adj_sample_spa = sp.csr_matrix(adj_sample_spa)
    adj_bs_spe = sp.csr_matrix(adj_bs_spe)
    adj_bs_spa = sp.csr_matrix(adj_bs_spa)

    return features, y_train, y_test, train_mask, test_mask, adj_sample_spe, adj_sample_spa, adj_bs_spe, adj_bs_spa,label


def load_pre_data(data_name):
    '''read data'''
    pre = sio.loadmat(os.getcwd() + '/data/pre_data/' + data_name + '_pre.mat')

    labels_ori = pre['labels_ori']
    train_loc = pre['train_loc']
    test_loc = pre['test_loc']

    return labels_ori,train_loc,test_loc

def construct_feed_dict(support_sample_spa, support_bs_spa,
                        support_bs_spe, features, labels, labels_mask, placeholders, support_sample_spe):

    feed_dict = dict()
    feed_dict.update({placeholders['support_sample_spa']: support_sample_spa})
    feed_dict.update({placeholders['support_sample_spe']: support_sample_spe})
    feed_dict.update({placeholders['support_bs_spa']: support_bs_spa})
    feed_dict.update({placeholders['support_bs_spe']: support_bs_spe})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def normalize_adj(adj):
    """normalized adjacency matrix"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # The variable whose element is negative infinity becomes 0 in d_inv_sqrt
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_to_tuple(sparse_mx):
    """Convert a sparse matrix to a tuple representation"""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):  # judge if sparse_mx is list
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def pre_process_features(features):
    """Row normalized feature matrix and transformed into tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def pre_process_adj(adj):
    """The adjacency matrix of GCN model is preprocessed and transformed into tuple representation"""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def contrary_one_hot(label):
    '''Convert onehot tags to real tags'''
    size=len(label)
    label_ori=np.empty(size)
    for i in range(size):
        label_ori[i]=np.argmax(label[i])+1
    return label_ori

def save_result(data_name,oa,aa,kappa,num_band_seclection_now,band_loction,per_class_acc,train_time,test_time):
    '''Save the experimental results in the txt file'''
    write_content='\n'+data_name+'\n'+'oa:'+str(oa)+' aa:'+str(aa)+' kappa:'+str(kappa)+'\n'+'num_band_seclection:'+str(num_band_seclection_now)+'\n'+'band_loction:'+str(band_loction)+'\n'+'per_class_acc:'+str(per_class_acc)+'\n'+'train_time:'+str(train_time)+' test_time:'+str(test_time)+'\n'
    f = open(os.getcwd()+'/indian实验结果.txt','a')
    f.writelines(write_content)
    f.close()
from pre_data import *
from .one_dim_cnn import one_dim_cnn
from pre_data.sample_cuda import cuSample

class Get_graph_file(object):

    def __init__(self, data_name):
        self.data_name=data_name
        self.path_in = PATH_ori
        self.path_out = PATH_graph+self.data_name+'_graph.mat'
        self.data = []
        self.labels = []
        self.data_norm = []
        self.readData()
        
        self.num_label = int(np.max(self.labels))
        self.num_sample = int(np.shape(self.data_norm)[0]*np.shape(self.data_norm)[1])-np.sum(self.labels==0)
        #self.num_sample = int(np.shape(self.data_norm)[0] * np.shape(self.data_norm)[1])         #构建全图矩阵
        self.num_band = int(np.shape(self.data_norm)[2])
        self.features = np.zeros([self.num_sample, self.num_band])
        self.features_loc=np.zeros([2, self.num_sample])
        self.construct_feature_loc()
        
        self.adj_sample_spe = np.zeros([self.num_sample, self.num_sample])
        self.adj_sample_spa = np.zeros([self.num_sample, self.num_sample])
        self.construct_adj_sample()

        self.adj_bs_spe = np.zeros([self.num_band, self.num_band])
        self.adj_bs_spa = np.zeros([self.num_band, self.num_band])
        self.construct_adj_band()

        self.save_data()

    def readData(self):
        if self.data_name == 'Indian_pines':
            self.data = sio.loadmat(self.path_in+'Indian_pines_corrected.mat')['indian_pines_corrected']
            self.labels = sio.loadmat(self.path_in+'Indian_pines_gt.mat')['indian_pines_gt']
        elif self.data_name == 'PaviaU':
            self.data = sio.loadmat(self.path_in+'PaviaU.mat')['paviaU']
            self.labels = sio.loadmat(self.path_in+'PaviaU_gt.mat')['paviaU_gt']
        elif self.data_name == 'KSC':
            self.data = sio.loadmat(self.path_in+'KSC.mat')['KSC']
            self.labels = sio.loadmat(self.path_in+'KSC_gt.mat')['KSC_gt']
        elif self.data_name == 'Salinas':
            self.data = sio.loadmat(self.path_in+'Salinas_corrected.mat')['salinas_corrected']
            self.labels = sio.loadmat(self.path_in+'Salinas_gt.mat')['salinas_gt']
        elif self.data_name == 'washington':
            self.data = sio.loadmat(self.path_in+'washington.mat')['washington_datax']
            self.labels = sio.loadmat(self.path_in+'washington_gt.mat')['washington_labelx']
        elif self.data_name == 'Houston':
            self.data = sio.loadmat(self.path_in+'Houstondata.mat')['Houstondata']
            self.labels = sio.loadmat(self.path_in+'Houstonlabel.mat')['Houstonlabel']
        elif self.data_name == 'Houston_1':
            self.data = sio.loadmat(self.path_in+'hou_0.mat')['hou_data']
            self.labels = sio.loadmat(self.path_in+'hou_0_change_label.mat')['hou_label']
        self.data = np.float64(self.data )
        self.labels = np.array(self.labels).astype(float)

        self.data_norm = np.zeros(np.shape(self.data))
        for i in range(np.shape(self.data)[0]):
            for j in range(np.shape(self.data)[1]):
                self.data_norm[i,j,:] = preprocessing.normalize(self.data[i,j,:].reshape(1,-1))[0]
                
    def construct_feature_loc(self):
        
        index=0
        for i in range(np.shape(self.data_norm)[0]):
            for j in range(np.shape(self.data_norm)[1]):
                if(self.labels[i][j]!=0):#构建全图矩阵
                    self.features[index,:]=self.data_norm[i,j,:]
                    self.features_loc[0][index]=i
                    self.features_loc[1][index]=j
                    index=index+1

    def construct_adj_sample(self):

        # for i in range(self.num_sample):
        #     for j in range(i,self.num_sample):
        #
        #         # adj_sample_spa
        #         if mean_squared_error(self.features_loc[:, i], self.features_loc[:, j]) < pow((NUM_adj_sample_spa_window / 2), 2):
        #             self.adj_sample_spa[i][j] = 1
        #             self.adj_sample_spa[j][i] = self.adj_sample_spa[i][j]
        #
        #         # 无监督adj_sample_spe
        #         if np.exp(-np.sqrt(mean_squared_error(self.features[i], self.features[j]))) > NUM_adj_sample_spe_threshold:
        #             self.adj_sample_spe[i][j] = np.exp(-np.sqrt(mean_squared_error(self.features[i], self.features[j])))
        #             self.adj_sample_spe[j][i] = self.adj_sample_spe[i][j]
        #
        #     # 保留top NUM_adj_sample_spe_retain个样本连接
        #     b = abs(copy.deepcopy(self.adj_sample_spe[i]))
        #     b.sort(axis=0)
        #     self.adj_sample_spe[i] = np.where(self.adj_sample_spe[i] > b[-(NUM_adj_sample_spe_retain + 1)],self.adj_sample_spe[i], 0)
        #     if i % 1000 == 0:
        #         print("%d samples have been converted." % i)
        import time
        t1 = time.clock()
        self.adj_sample_spa, self.adj_sample_spe = cuSample(self.features, self.features_loc, NUM_adj_sample_spa_window, NUM_adj_sample_spe_threshold,
                            NUM_adj_sample_spe_retain)
        print(time.clock()-t1)
        print("adj_sample is constructed!")

    def construct_adj_band(self):
        # adj_sample_spa
        for i in range(self.num_band):
            for j in range(i, self.num_band):
                self.adj_bs_spa[i][j] = np.exp(abs(i-j))
                self.adj_bs_spa[j][i] = self.adj_bs_spa[i][j]

        # adj_sample_spe
        for i in range(self.num_band):
            for j in range(i, self.num_band):
                self.adj_bs_spe[i][j] = np.exp(-np.sqrt(mean_squared_error(self.features[:, i], self.features[:, j])))
                self.adj_bs_spe[j][i] = self.adj_bs_spe[i][j]

        print("adj_bs is constructed!")

    def save_data(self):
        
        self.adj_sample_spe = sp.csr_matrix(self.adj_sample_spe)
        self.adj_sample_spa = sp.csr_matrix(self.adj_sample_spa)
        sio.savemat(self.path_out, {'features': self.features,
                                    'adj_sample_spe': self.adj_sample_spe,
                                    'adj_sample_spa': self.adj_sample_spa,
                                    'adj_bs_spe': self.adj_bs_spe,
                                    'adj_bs_spa': self.adj_bs_spa})

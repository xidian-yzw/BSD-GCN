# -*- coding: utf-8 -*-
from pre_data import *


class Get_pre_file(object):
    
    def __init__(self,data_name,ratio):
        self.data_name=data_name
        self.ratio=ratio
        self.path_in = PATH_ori
        self.path_out = PATH_pre+self.data_name+'_pre.mat'
        
        self.data = []
        self.labels = []
        self.readData()
        
        self.train_loc = []
        self.test_loc = []
        self.contruct_loc()
        
        self.num_label = int(np.max(self.labels))
        self.num_row=np.shape(self.data)[1]
        self.num_sample = int(np.shape(self.data)[0]*np.shape(self.data)[1])-np.sum(self.labels==0)
        #self.num_sample = int(np.shape(self.data)[0] * np.shape(self.data)[1])##构建全图矩阵
        self.num_band = int(np.shape(self.data)[2])
        self.y_train = np.zeros([self.num_sample, self.num_label])
        self.y_test = np.zeros([self.num_sample, self.num_label])
        self.train_mask = np.zeros([self.num_sample], dtype=bool)
        self.test_mask = np.zeros([self.num_sample], dtype=bool)
        self.construct_label_mask()
        
        self.save()
        
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
    
    def contruct_loc(self):
        ''' 从所有类中每类选取训练样本和测试样本 '''
        self._c = int(self.labels.max())
        self._x_loc1 = []
        self._x_loc2 = []
        self._y_loc1 = []
        self._y_loc2 = []
        for i in range(1, self._c+1):
        #i = 1
            self._loc1, self._loc2 = np.where(self.labels == i)
            self._num = len(self._loc1)
            self._order = np.random.permutation(range(self._num))
            self._loc1 = self._loc1[self._order]
            self._loc2 = self._loc2[self._order]
            self._num1 = int(np.round(self._num*self.ratio))
            self._x_loc1.extend(self._loc1[:self._num1])
            self._x_loc2.extend(self._loc2[:self._num1])
            self._y_loc1.extend(self._loc1[self._num1:])
            self._y_loc2.extend(self._loc2[self._num1:])
            self.train_loc = np.vstack([self._x_loc1, self._x_loc2])
            self.test_loc = np.vstack([self._y_loc1, self._y_loc2])
            
    def construct_label_mask(self):

        for i in range(self.num_sample):
            if i < np.shape(self.train_loc)[1]:
                index=self.train_loc[0][i]*self.num_row+self.train_loc[1][i]-np.sum(self.labels[0:self.train_loc[0][i]]==0)-np.sum(self.labels[self.train_loc[0][i]][:self.train_loc[1][i]]==0)
                label_index=int(self.labels[self.train_loc[0][i]][self.train_loc[1][i]] - 1)
                self.y_train[index][label_index] = 1
                self.train_mask[index] = True
            else:
                temp=i - np.shape(self.train_loc)[1]
                index=self.test_loc[0][temp]*self.num_row+self.test_loc[1][temp]-np.sum(self.labels[0:self.test_loc[0][temp]]==0)-np.sum(self.labels[self.test_loc[0][temp]][:self.test_loc[1][temp]]==0)
                label_index=int(self.labels[self.test_loc[0][temp]][self.test_loc[1][temp]] - 1)
                self.y_test[index][label_index] = 1
                self.test_mask[index] = True
    #构建全图矩阵
    # def construct_label_mask(self):
    #
    #     for i in range(len(self.train_loc)):
    #         index = self.train_loc[0][i] * self.num_row + self.train_loc[1][i]
    #         label_index = int(self.labels[self.train_loc[0][i]][self.train_loc[1][i]] - 1)
    #         self.y_train[index][label_index] = 1
    #         self.train_mask[index] = True
    #
    #     for i in range(len(self.test_loc)):
    #         index = self.test_loc[0][i] * self.num_row + self.test_loc[1][i]
    #         label_index = int(self.labels[self.test_loc[0][i]][self.test_loc[1][i]] - 1)
    #         self.y_test[index][label_index] = 1
    #         self.test_mask[index] = True

    def save(self):
        sio.savemat(self.path_out, {'y_train':self.y_train,
                                    'y_test':self.y_test,
                                    'train_mask':self.train_mask, 
                                    'test_mask':self.test_mask,
                                    'label':self.labels})


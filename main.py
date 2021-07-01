import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from model.model import GCN
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load data
data_set = "Indian_pines"  # data set name: 'Indian_pines' 'PaviaU' 'Houston'
ratio=0.05  # proportion of training samples
pre_data(data_set, ratio)  # produce pre and graph file
features, y_train, y_test, train_mask, test_mask, adj_sample_spe, adj_sample_spa, adj_bs_spe, adj_bs_spa,labels_ori = load_data(data_set)  # 载入数据
num_sample=np.shape(features)[0]

# pre_processing
features = pre_process_features(features)
support_sample_spa = pre_process_adj(adj_sample_spa)
support_sample_spe = pre_process_adj(adj_sample_spe)

support_bs_spa = pre_process_adj(adj_bs_spa)
support_bs_spe = pre_process_adj(adj_bs_spe)


# Settings
step = 0
global_step=tf.Variable(step)
learning_rate = tf.train.exponential_decay(0.001, global_step,50,0.8, staircase=True)
epochs = 1000

# Define placeholders
placeholders = {
    'support_sample_spa': tf.sparse_placeholder(tf.float32),
    'support_sample_spe': tf.sparse_placeholder(tf.float32),
    'support_bs_spa': tf.sparse_placeholder(tf.float32),
    'support_bs_spe': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
model = GCN(learning_rate,global_step, placeholders,sample_num=features[2][0], input_dim=features[2][1], epoch=epochs)
########################################################################################################################
# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

t_train = time.time()
# Train model
for epoch in range(epochs):
    # Training step

    feed_dict = construct_feed_dict(support_sample_spa, support_bs_spa, support_bs_spe,
                                    features, y_train, train_mask, placeholders, support_sample_spe)

    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    step+=1
    # Validation
    if epoch%20==0:
        feed_dict_val = construct_feed_dict(support_sample_spa, support_bs_spa, support_bs_spe,
                                           features, y_test, test_mask, placeholders, support_sample_spe)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]),
              "val_loss=", "{:.5f}".format(outs_val[0]),
              "val_acc=", "{:.5f}".format(outs_val[1]))
    # if epoch % 5 == 0:
    #     a=sess.run([model.mask_bs_], feed_dict=feed_dict)
    #     band_index = np.where( a[0] != 0)[0]  # indexes of selected bands
    #     write_content='\n'+data_set+'\n'+'band_index:'+str(band_index)
    #     f = open(os.getcwd()+'/band_index.txt','a')
    #     f.writelines(write_content)
    #     f.close()

time_train=time.time()-t_train
print("Optimization Finished!")

# Testing
t_test = time.time()
feed_dict_val = construct_feed_dict(support_sample_spa, support_bs_spa, support_bs_spe,
                                    features, y_test, test_mask, placeholders, support_sample_spe)
outs_val = sess.run([model.outputs_y,model.mask_bs_], feed_dict=feed_dict_val)
test_index=np.where(test_mask[0])
y_pr=contrary_one_hot(outs_val[0][test_index])
y_real=contrary_one_hot(y_test[test_index])

oa=accuracy_score(y_real,y_pr)
per_class_acc=recall_score(y_real,y_pr,average=None)
aa=np.mean(per_class_acc)
kappa=cohen_kappa_score(y_real,y_pr)
time_test=time.time()-t_test
band_index = np.where(outs_val[1] != 0)[0]  # indexes of selected bands

# save results
num_band=60
save_result(data_set,oa,aa,kappa,num_band,band_index,per_class_acc,time_train,time_test)

plot_max = np.zeros(np.shape(labels_ori))
index=0
predient_label=contrary_one_hot(outs_val[0])
for i in range(np.shape(plot_max)[0]):
    for j in range(np.shape(plot_max)[1]):
        if(labels_ori[i][j]!=0):
            plot_max[i][j] = predient_label[index]
            index+=1
sio.savemat(os.getcwd() + '/result/plot/' + data_set + '_' + 'plot.mat', {'plot_max': plot_max})
########################################################################################################################
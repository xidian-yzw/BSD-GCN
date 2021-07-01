# -*- coding: utf-8 -*-
from pre_data import *

class one_dim_cnn(object):
    def __init__(self,data_name,adj_sample_spe):
        self.name=data_name
        self.num_classification = 2

        self.x_train=[]
        self.y_train=[]
        self.x_test=[]
        self.y_test=[]
        self.features =[]
        self.labels = []
        self.train_loc_x= []
        self.train_loc_y = []
        self.test_loc_x = []
        self.test_loc_y = []
        self.load_data(self.name)

        self.num_layer=1
        self.vars = {}
        self.dim_input=np.shape(self.features)[1]

        self.activations = []  # 存储每层激活值
        self.outputs = None  # 存储输出值
        self.y_=None
        self.loss = 0  # 存储损失
        self.correct_prediction=None
        self.accuracy = 0  # 存储精度
        self.opt_op = None  # 优化操作
        self.init = None

        self.x_in = tf.placeholder(tf.float32, [None, self.dim_input, 2])
        self.y_out = tf.placeholder(tf.float32, [None, self.num_classification])
        self.adj_sample_spe = adj_sample_spe

        self.build()
        self.train_test_get_adj()

    def load_data(self,data_name):

        pre = sio.loadmat(PATH_pre + self.name + '_pre.mat')
        self.x_train = pre['train_x']
        self.y_train = pre['train_y'][0]
        self.x_test = pre['test_x']
        self.y_test = pre['test_y'][0]

        self.features = np.concatenate((self.x_train, self.x_test), axis=0)
        self.labels = np.concatenate((self.y_train, self.y_test), axis=0)

        self.train_loc_x = [i for i in range(len(self.x_train))]
        self.train_loc_y = [i for i in range(len(self.x_train))]
        self.test_loc_x = [i for i in range(len(self.features))]
        self.test_loc_y = [i for i in range(len(self.features))]

        np.random.shuffle(self.train_loc_x)
        np.random.shuffle(self.train_loc_y)
        np.random.shuffle(self.test_loc_x)
        np.random.shuffle(self.test_loc_y)

    def one_dim_cnn_layer(self, layer_input, kenel_size, input_dim, output_dim, act=tf.nn.relu):

        self.vars['weights'+ str(self.num_layer)] = tf.Variable(tf.truncated_normal([kenel_size, input_dim, output_dim], stddev=0.1))
        self.vars['bias'+ str(self.num_layer)] = tf.Variable(tf.constant(0.1, shape=[output_dim]))

        x = tf.nn.conv1d(layer_input, self.vars['weights'+ str(self.num_layer)], CNN_stride, padding='SAME')
        h = tf.nn.bias_add(x, self.vars['bias'+ str(self.num_layer)])
        bn = tf.contrib.layers.batch_norm(h, decay=0.9, epsilon=1e-5, scale=True, is_training=True)
        convout = act(bn)

        self.num_layer += 1

        return convout

    def build(self):

        self.activations.append(self.one_dim_cnn_layer(self.x_in, 3, 2, 32))
        self.activations.append(self.one_dim_cnn_layer(self.activations[-1], 3, 32, 64))
        self.activations.append(self.one_dim_cnn_layer(self.activations[-1], 3, 64, 64))
        self.activations.append(self.one_dim_cnn_layer(self.activations[-1], 3, 64, 32))

        self.activations.append(tf.reshape(self.activations[-1], [-1, 200 * 32]))

        self.vars['weights' + str(self.num_layer)] = tf.Variable(
            tf.truncated_normal([200 * 32, self.num_classification], stddev=0.1))
        self.vars['bias' + str(self.num_layer)] = tf.Variable(tf.constant(0.1, shape=[self.num_classification]))

        self.outputs=tf.add(tf.matmul(self.activations[-1], self.vars['weights' + str(self.num_layer)]),
                            self.vars['bias' + str(self.num_layer)])
        self.y_ = tf.nn.softmax(self.outputs)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_out, logits=self.outputs))
        self.opt_op = tf.train.RMSPropOptimizer(Learn_rate).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.y_out, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

    def train_test_get_adj(self):
        self.init = tf.global_variables_initializer()
        print("One-dimensional CNN training...")
        with tf.Session() as sess:
            sess.run(self.init)
            epoch = 0
            while epoch < Num_epoch:
                # display
                print('epoch %d' % (epoch), '/%d' % (Num_epoch))

                for i in range(len(self.train_loc_x)):
                    # next batch
                    batch_x=[np.concatenate([self.features[self.train_loc_x[i]][:, np.newaxis],
                                             self.features[self.train_loc_y[j]][:, np.newaxis]], axis=1)
                             for j in range(len(self.train_loc_y))]
                    batch_y_temp = [1 if self.labels[self.train_loc_x[i]] == self.labels[self.train_loc_y[j]] else 0
                                    for j in range(len(self.train_loc_y))]
                    batch_y = np.zeros([len(batch_y_temp), self.num_classification])
                    for j in range(len(batch_y_temp)):
                        batch_y[j, int(batch_y_temp[j])] = 1

                    # optimization
                    sess.run(self.opt_op, feed_dict={self.x_in: batch_x, self.y_out: batch_y})

                    print('step %d' % (i), '/%d' % (len(self.train_loc_x)))

                # update epoch
                epoch = epoch + 1
            print("Optimization Finished!")
            
            num = len(self.test_loc_x)
            accuracy=[]
                
            for i in range(num):
                # 构建测试数据
                test_input = [np.concatenate([self.features[self.test_loc_x[i]][:, np.newaxis],
                                              self.features[self.test_loc_y[j]][:, np.newaxis]], axis=1)
                              for j in range(len(self.test_loc_y))]
                        
                y_pred = self.y_.eval(feed_dict={self.x_in: test_input})
                for j in range(len(self.test_loc_y)):
                    self.adj_sample_spe[self.test_loc_x[i]][self.test_loc_y[j]] = y_pred[j][1]
                print('step %d' % (i), '/%d' % (num))
            print("Original adj_sample_spe is constructed!")

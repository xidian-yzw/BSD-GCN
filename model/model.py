from .layers import *


class GCN(object):

    def __init__(self,
                 learning_rate,
                 global_step,
                 placeholders,
                 sample_num,
                 input_dim,
                 epoch):

        self.name = self.__class__.__name__.lower()

        self.inputs = placeholders['features']
        self.sample_num = sample_num
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.learning_rate = learning_rate
        self.global_step = global_step
        self.epoch = epoch

        self.activations_n = []  # 存储每层激活值
        self.activations_y = []  # 存储每层激活值
        self.outputs_n = None  # 存储输出值
        self.outputs_y = None  # 存储输出值
        self.loss = 0  # 存储损失
        self.accuracy = 0  # 存储精度
        self.opt_op = None  # 优化操作

        self.build()

    def build(self):

        # Bs_mask_layer
        mask_layer = Layer(input_dim=self.sample_num,
                           output_dim=self.input_dim,
                           placeholders=self.placeholders,
                           bias=True,
                           dropout=False,
                           act=lambda x: x,
                           sparse_inputs=True,
                           name='Bs_mask_layer')
        self.mask_bs, self.mask_bs_ = mask_layer(self.inputs)

        # Bs_layer
        bs_layer = Layer(input_dim=self.input_dim,
                         output_dim=LAYER_ouput_dim,
                         placeholders=self.placeholders,
                         bias=False,
                         dropout=False,
                         act=tf.nn.relu,
                         sparse_inputs=True,
                         name='Bs_layer')
        self.activations_n.append(bs_layer(self.inputs, self.mask_bs))
        self.activations_y.append(bs_layer(self.inputs, self.mask_bs_))

        # block and dense
        for i in range(0,NUM_block):
            temp_n = self.activations_n[0]
            temp_y = self.activations_y[0]
            for ele in range(1, len(self.activations_n)):
                temp_n = tf.add(temp_n, self.activations_n[ele])
                temp_y = tf.add(temp_y, self.activations_y[ele])
            out_block=self.block(temp_n,temp_y,LAYER_ouput_dim*2,LAYER_ouput_dim)

            self.activations_n.append(out_block[0])
            self.activations_y.append(out_block[1])

        # Last_layer
        last_layer = Layer(input_dim=LAYER_ouput_dim*2,
                           output_dim=self.output_dim,
                           placeholders=self.placeholders,
                           bias=True,
                           dropout=False,
                           act=lambda x: x,
                           sparse_inputs=False,
                           name='Last_layer')
        self.outputs_n = last_layer(self.activations_n[-1])
        self.outputs_y = last_layer(self.activations_y[-1])

        # train and evaluate
        self._loss()
        self._accuracy()
        self.opt_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)

    def block(self, in_input_n, in_input_y, input_block_dim, output_block_dim):
        # Fc_layer
        # fir = Layer(input_dim=input_block_dim,
        #             output_dim=input_block_dim,
        #             placeholders=self.placeholders,
        #             bias=True,
        #             dropout=True,
        #             act=tf.nn.relu,
        #             sparse_inputs=False,
        #             mask_bs=None,
        #             name='Fc_layer')

        # Basic_gcn_layer
        sec = Layer(input_dim=input_block_dim,
                    output_dim=output_block_dim,
                    placeholders=self.placeholders,
                    bias=True,
                    dropout=True,
                    act=tf.nn.relu,
                    sparse_inputs=False,
                    name='Basic_gcn_layer')

        # Fc_layer
        # thr = Layer(input_dim=output_block_dim,
        #             output_dim=output_block_dim,
        #             placeholders=self.placeholders,
        #             bias=True,
        #             dropout=True,
        #             act=tf.nn.relu,
        #             sparse_inputs=False,
        #             mask_bs=None,
        #             name='Fc_layer')

        # fir_out = fir(in_input)
        sec_out_n = sec(in_input_n)
        sec_out_y = sec(in_input_y)
        # thr_out = thr(sec_out)

        return sec_out_n,sec_out_y

    def _loss(self):

        # Cross entropy error
        # self.ratio = tf.cast(self.global_step/self.epoch,dtype=tf.float32)
        self.ratio = tf.cast(self.epoch / self.epoch, dtype=tf.float32)
        # self.loss += (1-self.ratio)*masked_softmax_cross_entropy(self.outputs_n, self.placeholders['labels'],
        #                                                          self.placeholders['labels_mask'])
        self.loss += self.ratio*masked_softmax_cross_entropy(self.outputs_y, self.placeholders['labels'],
                                                             self.placeholders['labels_mask'])


    def _accuracy(self):

        # calculation accuracy
        self.accuracy = masked_accuracy(self.outputs_y, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

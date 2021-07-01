from .utils import *


class Layer(object):
    """ 图卷积层 """

    def __init__(self,
                 input_dim,
                 output_dim,
                 placeholders,
                 bias=False,
                 dropout=False,
                 act=tf.nn.relu,
                 sparse_inputs=False,
                 name=None):

        assert name in Allowed_kwargs, 'Invalid name argument: ' + name
        self.name = name
        self.scope = name + '_' + str(get_layer_uid(name))

        self.vars = {}  # 存储变量
        self.bias = bias  # 是否有偏值
        self.act = act  # 激活函数
        self.input_dim = input_dim
        self.output_dim=output_dim
        self.sparse_inputs = sparse_inputs  # 是否为稀疏数据
        self.dropout = NUM_dropout if dropout else 0  # 设置dropout率
        self.num_features_nonzero = placeholders['num_features_nonzero']  # 稀疏dropout辅助变量

        self.support_spe = placeholders['support_sample_spe']  # support是邻接矩阵的归一化形式
        self.support_spa = placeholders['support_sample_spa']
        self.support_bs_spe = placeholders['support_bs_spe']
        self.support_bs_spa = placeholders['support_bs_spa']

        with tf.variable_scope(self.scope + '_vars'):  # 构建并初始化权值和偏值
            self.vars['weights_spe'] = glorot([self.input_dim, self.output_dim], name='weights_spe')
            self.vars['weights_spa'] = glorot([self.input_dim, self.output_dim], name='weights_spa')
            if self.bias:
                self.vars['bias_spe'] = zeros([self.output_dim], name='bias_spe')
                self.vars['bias_spa'] = zeros([self.output_dim], name='bias_spa')

    def __call__(self, inputs, mask_bs= None):

        if self.name=='Basic_gcn_layer':
            return self.basic_gcn_layer(inputs)
        if self.name=='Bs_mask_layer':
            return self.bs_mask_layer(inputs)
        if self.name=='Bs_layer':
            return self.bs_layer(inputs, mask_bs)
        if self.name=='Last_layer':
            return self.last_layer(inputs)
        if self.name=='Fc_layer':
            return self.fc_layer(inputs)

    def basic_gcn_layer(self, inputs):
        with tf.name_scope(self.scope):

            # dropout
            if self.sparse_inputs:
                inputs = sparse_dropout(inputs, 1 - self.dropout, self.num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout)

            # convolve
            pre_sup_spe = dot(inputs, self.vars['weights_spe'], sparse=self.sparse_inputs)
            support_spe = dot(self.support_spe, pre_sup_spe, sparse=True)
            pre_sup_spa = dot(inputs, self.vars['weights_spa'], sparse=self.sparse_inputs)
            support_spa = dot(self.support_spa, pre_sup_spa, sparse=True)

            # bias
            if self.bias:
                support_spe += self.vars['bias_spe']
                support_spa += self.vars['bias_spa']

            # concat
            output = tf.concat([support_spe, support_spa], 1)

            # BN
            bn = tf.contrib.layers.batch_norm(output, decay=0.9, epsilon=1e-5, scale=True, is_training=True)

        return self.act(bn)

    def bs_mask_layer(self, inputs):
        with tf.name_scope(self.scope):

            # convolve
            pre_sup_spe = dot(tf.sparse_transpose(inputs), self.vars['weights_spe'], sparse=self.sparse_inputs)
            support_spe = dot(self.support_bs_spe, pre_sup_spe, sparse=True)
            pre_sup_spa = dot(tf.sparse_transpose(inputs), self.vars['weights_spa'], sparse=self.sparse_inputs)
            support_spa = dot(self.support_bs_spa, pre_sup_spa, sparse=True)

            # bias
            if self.bias:
                support_spe += self.vars['bias_spe']
                support_spa += self.vars['bias_spa']

            # add
            shayebushi =tf.diag_part(tf.add(support_spe,support_spa))
            self.output = (shayebushi-tf.reduce_min(shayebushi))/(tf.reduce_max(shayebushi)-tf.reduce_min(shayebushi))

            # top k
            top_k_val = tf.nn.top_k(tf.transpose(self.output), NUM_select_bs+1)[0][-1]
            zero = tf.zeros_like(self.output)
            self.outputs = tf.where(self.output > top_k_val, self.output, zero)

        return self.act(self.output), self.act(self.outputs)

    def bs_layer(self, inputs, mask_bs):
        with tf.name_scope(self.scope):

            # conduce the mask
            mask_bs_expand=tf.expand_dims(mask_bs,1)
            mask_bs_expand=tf.tile(mask_bs_expand,[1,self.output_dim])
            self.temp_wight_spe = tf.multiply(mask_bs_expand, self.vars['weights_spe'])
            self.temp_wight_spa = tf.multiply(mask_bs_expand, self.vars['weights_spa'])

            # convolve
            pre_sup_spe = dot(inputs, self.temp_wight_spe, sparse=self.sparse_inputs)
            output_spe = dot(self.support_spe, pre_sup_spe, sparse=True)
            pre_sup_spa = dot(inputs, self.temp_wight_spa, sparse=self.sparse_inputs)
            output_spa = dot(self.support_spa, pre_sup_spa, sparse=True)

            # concat
            output=tf.concat([output_spe, output_spa], 1)

        return self.act(output)

    def last_layer(self, inputs):

        with tf.name_scope(self.scope):

            # convolve
            pre_sup_spe = dot(inputs, self.vars['weights_spe'], sparse=self.sparse_inputs)
            support_spe = dot(self.support_spe, pre_sup_spe, sparse=True)
            pre_sup_spa = dot(inputs, self.vars['weights_spa'], sparse=self.sparse_inputs)
            support_spa = dot(self.support_spa, pre_sup_spa, sparse=True)

            # bias
            if self.bias:
                support_spe += self.vars['bias_spe']
                support_spa += self.vars['bias_spa']

            # add
            output=tf.add(support_spe, support_spa)

        return self.act(output)

    def fc_layer(self, inputs):
        with tf.name_scope(self.scope):

            # dropout
            if self.sparse_inputs:
                inputs = sparse_dropout(inputs, 1 - self.dropout, self.num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout)

            # fc
            pre_sup = dot(inputs, self.vars['weights_spe'], sparse=self.sparse_inputs)

            # bias
            if self.bias:
                pre_sup += self.vars['bias_spe']

            # BN
            bn = tf.contrib.layers.batch_norm(pre_sup, decay=0.9, epsilon=1e-5, scale=True, is_training=True)

        return self.act(bn)
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, position, bias=True,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 **kwargs
                 ):

        """
        Describe：
            构建Attention模块，分别可作用于模型的Embedding、Conv1D、替换Pooling；不同类型下，处理稍有差异。

        param: step_dim : 表示用于计算Attention的序列数据长，“感受野”
        param: position : 表示Attention作用的位置，可取"embedding"/"conv1d"/"pooling"
        param: bias     : True or False。表示是否给注意力矩阵增加偏置b

        param: W_regularizer：
        param: W_constraint：
        param: b_regularizer：
        param: b_constraint：
        """
        self.position = position
        self.bias = bias
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        # 特征维度（在NLP任务中表示Embedding的维度）
        self.features_dim = 0
        self.step_dim = 0
        """
        glorot_uniform:也称之为Xavier uniform initializer，由一个均匀分布（uniform distribution)来初始化数据。
        均匀分布的区间是[-limit, limit]
        其中,limit=sqrt(6 / (fan_in + fan_out));fan_in和fan_out分别表示输入单元的结点数和输出单元的结点数。
        """
        self.init = initializers.get('glorot_uniform')

        # 继承超类的参数初始化
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Describe:
            这个函数用来确立这个层都有哪些参数（可训练）。
            这个方法必须设self.built = True,以完成参数的build工作。
        Args:
            input_shape：用来指定输入张量的shape。然而这个input_shape只需要在模型的首层加以指定。
                         一旦模型的首层的input_shape指定了，后面会根据计算图自动推断。
                         PS：但是一般都加assert用来验证传的参数的shape对不对（From：吴恩达视频说的）
        """
        # 三种不同模式下的，input_shape含义
        # -- embedding: [sample,step,fetures] --
        # -- conv1d:    [sample,features,filters] --
        # -- pooling:   [sample,features,filters] --

        assert len(input_shape) == 3
        if self.position != "embedding":
            step_shape = input_shape[1]
            feature_shape = 1
        else:
            step_shape = input_shape[1]
            feature_shape = input_shape[2]

        self.features_dim = feature_shape
        self.step_dim = step_shape

        # 注意力矩阵的参数,列向量，行数等于（fetures）
        # 是由(1,fetures)*注意力参数(fetures,1)。得到的是1个对应于某一个step的注意力
        self.W = self.add_weight(shape=(feature_shape,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.bias:
            self.b = self.add_weight(shape=(step_shape,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def call(self, x, mask=None):
        """
        Describe:
            自定义Layer的核心层，用于计算逻辑的。
        """
        features_dim = self.features_dim
        step_dim = self.step_dim

        if self.position != "embedding":
            # [sample,feature,filters] -> [sample,filters,step,1]
            x = tf.transpose(x, perm=[0, 2, 1])
            x = tf.expand_dims(x, -1)
        else:
            # [sample,step,feature]
            pass
        e = K.dot(x, self.W)

        if self.position != "embedding":
            # [sample,filters,step,1] -> [sample,filters,step]
            e = K.reshape(e, (-1, x.shape[1], step_dim))
        else:
            # [sample,step,feature] -> [sample,step]
            e = K.reshape(e, (-1, step_dim))

        if self.bias:
            e += self.b
        e = K.tanh(e)
        a = K.exp(e)

        # TODO 如果要加mask的话，在这里加
        # a *= K.cast(mask, K.floatx())

        # -- 计算SoftMax -- 训练初期避免NaN,加上一个很小的的值：epsilon = 1 * 10^-7
        a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)

        c = None
        if self.position == "embedding":
            # c.shape = [sample,step,feature]
            c = a * x

        elif self.position == "conv1d":
            # c.shape = [sample,filters,step,1]
            c = a * x
            c = K.reshape(c, (-1, x.shape[1], step_dim))
            c = tf.transpose(c, perm=[0, 2, 1])

        elif self.position == "pooling":
            # c.shape = [sample,filters,1]
            c = K.sum(a * x, axis=-2, keepdims=False)
            c = tf.transpose(c, perm=[0, 2, 1])
        return c

    def compute_output_shape(self, input_shape):
        if self.position == "embedding":
            return input_shape[0], self.step_dim, self.features_dim
        elif self.position == "conv1d":
            return input_shape[0], self.step_dim, input_shape[2]
        elif self.position == "pooling":
            return input_shape[0], input_shape[2]

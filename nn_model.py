from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, \
    Concatenate, BatchNormalization, Activation, Dense
from nn_layers import Attention


class Att_1_TextCNN(object):
    def __init__(self, max_len, vocab_count, embedding_dims, cnn_kernel_sizes, cnn_filters_num,
                 dense1_units, label_count):
        """
        :param max_len          : 用户query的最大长
        :param vocab_count      : w2v的总词表长 + 2（含_PAD与_UNK）
        :param embedding_dims   : embedding的维度
        :param cnn_kernel_sizes : TextCNN的卷积核大小,给入的是一个list:[3,4,5]
        :param cnn_filters_num  : TextCNN的卷积核个数
        :param dense1_units     : 全连接层的单元数
        :param label_count      : 子任务的分类类别数,给入的是一个list:[Age_Count,Gender_Count,Education_Count]
        """
        self.max_len = max_len
        self.vocab_count = vocab_count
        self.embedding_dims = embedding_dims
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.cnn_filters_num = cnn_filters_num
        self.dense1_units = dense1_units
        self.label_count = label_count

    def get_model(self, embedding_matrix):
        """
        创建模型
        :param embedding_matrix : 预训练的w2v参数矩阵
        :return:
            model : 创建好的模型
        """
        input_query = Input((self.max_len,))

        embedding = Embedding(input_dim=self.vocab_count,
                              output_dim=self.embedding_dims,
                              weights=[embedding_matrix], trainable=False,
                              input_length=self.max_len)(input_query)

        att_embedding = Attention(position="embedding", bias=True)(embedding)

        convs = []
        for kernel_size in self.cnn_kernel_sizes:
            c = Conv1D(filters=self.cnn_filters_num, kernel_size=kernel_size)(att_embedding)
            c = BatchNormalization()(c)
            c = Activation(activation="relu")(c)
            c = GlobalMaxPool1D()(c)
            convs.append(c)

        concat_convs = Concatenate()(convs)

        x0 = Dense(self.dense1_units, activation="relu")(concat_convs)
        output_0 = Dense(self.label_count[0], activation="softmax", name="out_Age")(x0)

        x1 = Dense(self.dense1_units, activation="relu")(concat_convs)
        output_1 = Dense(self.label_count[1], activation="sigmoid", name="out_Gender")(x1)

        x2 = Dense(self.dense1_units, activation="relu")(concat_convs)
        output_2 = Dense(self.label_count[2], activation="softmax", name="out_Education")(x2)

        model = Model(input=input_query, outputs=[output_0, output_1, output_2])
        model.summary()
        return model

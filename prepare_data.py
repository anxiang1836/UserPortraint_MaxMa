import pandas as pd
import tensorflow as tf
import jieba
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


class Prepare:

    def __init__(self, data_path, w2vpath, max_len):
        """
        :param data_path: 原始CSV的数据表路径
        :param w2vpath:  word2vec的模型路径
        :param max_len: 每个用户的query总长
        """
        self.data_path = data_path
        self.w2vpath = w2vpath
        self.max_len = max_len

        # 单词的索引
        self._word2idx = {}

    def create_embedding_matrix(self):
        """
        根据word2vec的model，创建Embedding层的初始化参数矩阵，并将词表的索引赋值和成员变量self._word2idx

        :return:
            embedding_matrix:用于初始化的Embedding层的参数矩阵
        """

        # -- 词转为index的索引表 --
        w2v_model = Word2Vec.load(self.w2vpath)
        vocab_list = [(k, w2v_model.wv[k]) for k, v in w2v_model.wv.vocab.items()]
        word2idx = {"_PAD": 0, "_UNK": 1}
        embedding_matrix = np.zeros((len(vocab_list) + 2, w2v_model.vector_size))

        unk = np.random.random(size=w2v_model.vector_size)
        unk = unk - unk.mean()
        embedding_matrix[1] = unk

        # -- 构建embedding的初始化参数矩阵 --
        for i in range(len(vocab_list)):
            word = vocab_list[i][0]
            word2idx[word] = i + 2
            embedding_matrix[i + 2] = vocab_list[i][1]

        self._word2idx = word2idx

        return embedding_matrix

    def split_dataset(self):
        """
        划分数据集：取0.1作为测试集，0.72作为训练集，0.18作为验证集。
        其中，验证集分别按照Age，Gender，Education的分布，划分了三个不同的训练集、验证集，就算是做3套不同的参数，后面的模型增强用的吧！

        :return:
            train_ds_set、val_ds_set是分别包含3套不同的DataSet的,分别是：By_Age,By_Gender,By_Education;
            test_ds是1个DataSet。
        """
        data = pd.read_csv(self.data_path, encoding='utf-8')

        X_ID_train, X_ID_test, y_train, _ = train_test_split(data['ID'], data['Age'],
                                                             stratify=data['Age'], test_size=0.1)

        # -- 分别按照Age、Gender、Education的分布去划分训练集和验证集，有3套不同的<训练+验证数据>
        train_ds_set = []
        val_ds_set = []
        for label in ['Age', 'Gender', 'Education']:
            # 切分验证集
            X_data = data.loc[data['ID'].isin(X_ID_train), :]
            X_ID_train, X_ID_val, _, _ = train_test_split(X_ID_train, X_data[label],
                                                          stratify=y_train, test_size=0.2)

            # 构建数据管道
            train = data.loc[data['ID'].isin(X_ID_train), :]
            val = data.loc[data['ID'].isin(X_ID_val), :]

            train_ds = self.create_ds(train)
            val_ds = self.create_ds(val)

            train_ds_set.append(train_ds)
            val_ds_set.append(val_ds)

        test = data.loc[data['ID'].isin(X_ID_test), :]
        test_ds = self.create_ds(test)

        return train_ds_set, val_ds_set, test_ds

    def create_ds(self, data):
        """
        根据完整的数据表，创建tf的数据管道（为切分后的数据）

        :param data: 包含多列的DataFrame
        :return:
            ds :  tensorflow的DataSet，算是一种数据流，很好用哦，官方推荐的数据渠道
        """
        text_data = self.__transfer(data['Query_list'])
        text_ds = tf.data.Dataset.from_tensor_slices(text_data)
        label_ds = tf.data.Dataset.from_tensor_slices(
            (data['Age'] - 1,
             data['Gender'] - 1,
             data['Education'] - 1))
        ds = tf.data.Dataset.zip((text_ds, label_ds))
        return ds

    def __transfer(self, query_list):
        """
        :param query_list: 是原始数据中的data["Query_list"]的那一列数据
        :return:
        """
        word2idx = self._word2idx

        # -- 将训练数据转为对应的index + padding矩阵 --
        train_data = []
        for line in query_list:
            filter_r = r"[\s+\d+\.\!\/_,$%^*():：+\"\']+|[+——！，。？、~@#￥%……&*（）-]+|[“”\[\];?《》’【】■]+"
            line = re.sub(filter_r, '', line)

            word_list = jieba.lcut(line)
            word_list = [word for word in word_list if (word is not '\t') & (re.match(r'\s', word) is None)]
            word_list = word_list[:self.max_len]

            line_indexs = []

            for word in word_list:
                if word in word2idx.keys():
                    idx = word2idx[word.strip()]
                else:
                    idx = 1
                line_indexs.append(idx)

            line_indexs = np.array(line_indexs).reshape(1, len(line_indexs))
            line_indexs = tf.keras.preprocessing.sequence.pad_sequences(line_indexs, max_len=self.max_len)

            train_data.append(line_indexs)

        train_data = np.array(train_data)
        train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[-1]))

        return train_data

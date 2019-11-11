import pandas as pd
import tensorflow as tf
import jieba
import re
import numpy as np
import pickle
import time
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence


class Prepare:

    def __init__(self, data_path, w2vpath, output_path, max_len,
                 word2idx=None):
        """
        :param data_path: 原始CSV的数据表路径
        :param w2vpath:  word2vec的模型路径
        :param max_len: 每个用户的query总长
        """
        self.data_path = data_path
        self.w2vpath = w2vpath
        self.output_path = output_path
        self.max_len = max_len

        # 单词的索引
        self._word2idx = word2idx

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
        # -- 持久化Embedding层的参数 --
        # self.__dump_obj(embedding_matrix, "embedding_matrix")
        # self.__dump_obj(word2idx, "word2idx")
        print("Embedding_matrix初始化完毕！Word2idx初始化完毕！")
        return embedding_matrix

    def split_dataset(self, label):
        """
        划分数据集：取0.1作为测试集，0.72作为训练集，0.18作为验证集。
        其中，验证集分别按照Age，Gender，Education的分布，可划分了三个不同的训练集、验证集

        :params label : 这个是按照哪个数据的分布对数据进行划分
        :return:
            train_ds_set、val_ds_set是分别包含3套不同的DataSet的,分别是：By_Age,By_Gender,By_Education;
            test_ds是1个DataSet。
        """
        data = pd.read_csv(self.data_path, encoding='utf-8')

        X_ID_train, X_ID_test, y_train, _ = train_test_split(data['ID'], data['Age'],
                                                             stratify=data['Age'], test_size=0.1)

        # 切分验证集
        X_data = data.loc[data['ID'].isin(X_ID_train), :]
        X_ID_train, X_ID_val, _, _ = train_test_split(X_ID_train, X_data[label],
                                                      stratify=X_data[label], test_size=0.2)

        # 构建数据管道
        train = data.loc[data['ID'].isin(X_ID_train), :]
        train_count = train.shape[0]
        val = data.loc[data['ID'].isin(X_ID_val), :]
        val_count = val.shape[0]
        test = data.loc[data['ID'].isin(X_ID_test), :]
        test_count = test.shape[0]

        data_dic = {"train": train, "val": val, "test": test}
        ds_set = []

        for k in data_dic.keys():
            print("开始创建{}分布划分的{}数据集".format(label, k))
            csv_data = data_dic[k]
            text_data = self.__transfer(data_dic[k]["Query_list"])
            ds = self.create_ds(csv_data, text_data)
            ds_set.append(ds)
            # 持久化
            self.__dump_obj(csv_data, k)
            self.__dump_obj(text_data, "{}_text_data".format(k))

        return ds_set, train_count, val_count, test_count

    def __dump_obj(self, obj, name):
        """
        将对象持久化到硬盘
        :param obj:待持久化的对象
        :param name:持久化的文件名称
        :return:
            None
        """
        timestamp = time.strftime("%m%d_%H%M", time.localtime())
        pickle.dump(obj, open(self.output_path + timestamp + '_' + name + '.pkl', 'wb'))
        return

    def __transfer(self, query_list):
        """
        用于转化原始数据，根据w2v的词表，转化为idx矩阵

        :param query_list: 是原始数据中的data["Query_list"]的那一列数据
        :return:
        """
        word2idx = self._word2idx

        # -- 将训练数据转为对应的index + padding矩阵 --
        train_data = []
        for i, line in enumerate(query_list):
            filter_r = r"[\s+\d+\.\!\/_,$%^*():：+\"\']+|[+——！，。？、~@#￥%……&*（）-]+|[“”\[\];?《》’【】■]+"
            line = re.sub(filter_r, '', line)

            word_list = jieba.lcut(line)
            word_list = [word for word in word_list if (word is not '\t') & (re.match(r'\s', word) is None)]
            word_list = word_list[:self.max_len]
            if i % 10000 == 0:
                print("第{}条query分词结果的前15个词{}：".format(i, word_list[:15]))

            line_indexs = []

            for word in word_list:
                if word in word2idx.keys():
                    idx = word2idx[word.strip()]
                else:
                    idx = 1
                line_indexs.append(idx)

            line_indexs = np.array(line_indexs).reshape(1, len(line_indexs))
            line_indexs = sequence.pad_sequences(line_indexs, maxlen=self.max_len)

            train_data.append(line_indexs)

        train_data = np.array(train_data)
        train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[-1]))

        return train_data

    @staticmethod
    def create_ds(df, text_data):
        """
        根据完整的数据表，创建tf的数据管道（为切分后的数据）

        :param data: 包含多列的DataFrame
        :return:
            ds :  tensorflow的DataSet，算是一种数据流，很好用哦，官方推荐的数据渠道
        """
        # text_data = self.__transfer(data['Query_list'])
        text_ds = tf.data.Dataset.from_tensor_slices(text_data)
        label_ds = tf.data.Dataset.from_tensor_slices(
            (df['Age'] - 1,
             df['Gender'] - 1,
             df['Education'] - 1))
        ds = tf.data.Dataset.zip((text_ds, label_ds))
        return ds

import pandas as pd
import jieba
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class PrepareData(object):
    def __init__(self,
                 input_csv_path,
                 output_csv_path,
                 output_tfidf_path,
                 stop_word_path
                 ):
        """
        :input_csv_path    : CSV表的输入路径
        :output_csv_path   : CSV表的输出路径
        :output_tfidf_path : tfidf-Vectorizer的持久化路径
        :stop_word_path    : 停用词表路径
        :output_w2v_path   : [弃用状态]
        """
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.output_tfidf_path = output_tfidf_path
        self.stop_word_path = stop_word_path
        self.output_w2v_path = ""

        self._data = pd.read_csv(self.input_csv_path, sep="###__###", header=None, encoding='utf-8')
        self._data.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_list']

        file = open(self.stop_word_path, encoding='utf-8')
        stop_words_list = file.readlines()
        file.close()
        self.stopwords = [word.strip() for word in stop_words_list]


# ----------重写数据-----------------
    def read_and_rewrite(self):
        data = self._data
        # -- 每个人的queryList统计量计算 --

        # 统计每个人的query数量
        data['Query_count'] = data['Query_list'].map(lambda x: len(x.split('\t')))
        # 统计每个人的包含空格的query个数
        data['Query_has_space'] = data['Query_list'].map(self.__count_query_has_space)
        print("数据统计量计算完毕！")

        # -- 清洗数据 --
        data['Query_list'] = data['Query_list'].map(self.__clean_query)
        print("数据清洗完毕！！")

        # -- 补全缺失值 --
        tfv = TfidfVectorizer(tokenizer=self.__Tokenizer(stopwords=self.stopwords), min_df=3, max_df=0.95,
                              sublinear_tf=True)
        X_tfidf = tfv.fit_transform(data['Query_list'])
        pickle.dump(X_tfidf, open(self.output_tfidf_path, 'wb'))  # 持久化
        print("tfidf-vector持久化完毕！")

        for label in ['Age', 'Gender', 'Education']:
            train_data = data.loc[data[label] != 0, label]
            to_befilled_data = data.loc[data[label] == 0, label]

            train_index = train_data.index
            to_befilled_index = to_befilled_data.index

            clf = LogisticRegression(C=2)
            clf.fit(X_tfidf[train_index], train_data)

            data[label][to_befilled_index] = clf.predict(X_tfidf[to_befilled_index])

        self._data = data
        print('数据补全完毕！')

        # 持久化数据表
        data.to_csv(self.output_csv_path, index=None, encoding='utf-8')
        return


    # --------准备W2v训练数据---------------
    # def prepare_for_w2v(self):
    #     pass
    #     return

    # ----------数据清洗静态方法-------------
    @staticmethod
    def __clean_query(query_list):
        new_list = []
        for query in query_list.split('\t'):
            # 删除带空格的query
            if re.match(r'https?://', query.strip()) is not None:
                continue
            # 删除掉所有数字
            query = re.sub(r'\d', '', query.strip())
            if len(query) != 0:
                new_list += query
            # 替换所有的公式
            # TODO 找到公式替换规则（数据发现再去看）
        return '\t'.join(new_list)


    # ----------统计空格率静态方法-----------
    @staticmethod
    def __count_query_has_space(query_list):
        count = 0
        query_list = query_list.split('\t')
        for query in query_list:
            if re.match(r'\s', query.strip()) is not None:
                count += 1
        # 返回一个比例，即有space的query占全部总query数的比例
        return count / len(query_list)


    class __Tokenizer:
        def __init__(self, stopwords):
            self.n = 0
            self.stopwords = stopwords

        def __call__(self, line):
            tokens = []
            for query in line.split('\t'):
                word_list = jieba.lcut(query)
                # 去停用词
                word_list = [word for word in word_list if word not in self.stopwords]

                for n_gram in [1, 2]:
                    for i in range(len(word_list) - n_gram + 1):
                        tokens += ['_*_'.join(word_list[i:i + n_gram])]
            return tokens


if __name__ == '__main__':
    params = {'input_csv_path': './data/train.csv',
              'output_csv_path': './data/new_train_20191110.csv',
              'output_tfidf_path': './data/tfidf/vector_20191110.pkl',
              'stop_word_path': './data/stopwords_hit.txt'}

    preprocess = PrepareData(**params)
    preprocess.read_and_rewrite()

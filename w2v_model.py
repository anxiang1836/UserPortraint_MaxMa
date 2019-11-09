from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import multiprocessing


class TrainWord2Vec:
    def __init__(self,
                 train_text_path,
                 output_w2vModel,
                 input_w2vModel=None,
                 incremental=False):
        self.train_text_path = train_text_path
        self.output_w2vModel = output_w2vModel
        self.input_w2vModel = input_w2vModel
        self.incremental = incremental

        train_file = open(self.train_text_path, 'r', encoding='utf-8')
        self._train_data = train_file.readlines()
        train_file.close()

    # -- 增量训练 或 常规从头训练 --
    def train(self, size=128, window=5, sg=0, hs=0, negative=3):
        # - 常规训练 -
        if self.incremental is False:
            model = Word2Vec(LineSentence(self._train_data),
                             size=size, window=window,
                             sg=sg, hs=hs, negative=negative,
                             workers=multiprocessing.cpu_count())

        # 异常,增量训练但未给定已有词向量
        elif (self.incremental is True) & (self.input_w2vModel is None):
            raise Exception("param input_w2vModel is None")

        # - 增量训练 -
        elif (self.incremental is True) & (self.input_w2vModel is not None):
            model = Word2Vec.load(self.input_w2vModel)
            print("现有w2v模型词表长为：{}".format(model.corpus_count))

            model.build_vocab(LineSentence(self._train_data))
            model.train(LineSentence(self._train_data),total_examples=model.corpus_count,epochs=model.iter)

        model.save(self.output_w2vModel)
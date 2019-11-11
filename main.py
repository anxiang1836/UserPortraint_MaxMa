import pickle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns

from nn_model import Att_1_TextCNN
from prepare_data import Prepare

if __name__ == "__main__":
    # --- 1.准备训练数据 ---
    word2idx = None
    # word2idx_path = ""
    # word2idx = pickle.load(open(word2idx_path, "rb"))

    BATCH_SIZE = 128

    params_for_prepare = {
        "data_path": "./data/new_train_20191110.csv",
        "w2vpath": "./word2vec_trained/baike_26g_news_13g_novel_229g.model",
        "output_path": "./prepare_data/",
        "max_len": 650,
        "word2idx": word2idx}

    prepare = Prepare(**params_for_prepare)
    embedding_martix = prepare.create_embedding_matrix()

    ds_set, train_count, val_count, test_count = prepare.split_dataset("Age")
    train_ds = ds_set[0]
    val_ds = ds_set[1]

    # train_path = ""
    # train_text_path = ""
    # val_path = ""
    # val_text_path = ""
    # test_path = ""
    # test_ds_path = ""
    #
    # train = pickle.load(open(train_path,"rb"))
    # train_text = pickle.load(open(train_text_path,"rb"))
    # val = pickle.load(open(val_path,"rb"))
    # val_text = pickle.load(open(val_text_path,"rb"))
    # test = pickle.load(open(test_path,"rb"))
    # test_text = pickle.load(open(test_ds_path, "rb"))
    # train_ds = prepare.create_ds(train,train_text)
    # val_ds = prepare.create_ds(val,val_text)

    train_ds = train_ds.shuffle(buffer_size=train_count).repeat(-1)
    train_ds = train_ds.batch(BATCH_SIZE)

    train_steps = train_count // BATCH_SIZE
    val_steps = val_count // BATCH_SIZE

    # --- 2.编译模型 ---
    EPOCHS = 50
    params_for_model = {
        "max_len": 650,
        "vocab_count": embedding_martix.shape[0],
        "embedding_dims": 128,
        "cnn_kernel_sizes": [3, 4, 5],
        "cnn_filters_num": 256,
        "dense1_units": 128,
        "label_count": [6, 2, 6]
    }
    adam_opt = Adam()
    model = Att_1_TextCNN(**params_for_model).get_model(embedding_martix)
    model.compile(optimizer=adam_opt,
                  loss={'out_Age': 'sparse_categorical_crossentropy',
                        'out_Gender': 'binary_crossentropy',
                        "out_Education": "sparse_categorical_crossentropy"},
                  metrics=['acc']
                  )

    # --- 3.训练模型 ---
    early_stopping = EarlyStopping(monitor="val_acc", patience=3, mode="max")
    history = model.fit(train_ds,
                        callbacks=[early_stopping],
                        epochs=EPOCHS,
                        steps_per_epoch=train_steps,
                        validation_data=val_ds,
                        validation_steps=val_steps)

    # --- 4.可视化训练 ---
    # step4:可视化一下模型
    print(history.history.keys())
    # plt.plot(history.epoch, history.history.get('acc'), label='acc')
    # plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
    # plt.legend()

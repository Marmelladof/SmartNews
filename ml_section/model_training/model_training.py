import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow.python.keras as keras
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# SPLITTING
def split_data(dataframe, train_ration, val_ration):
    _true = dataframe.loc[dataframe.Fake == 'True']
    _false = dataframe.loc[dataframe.Fake == 'False']
    chunks = {"train": [], "val": [], "test": []}
    length = len(dataframe.index)
    train_stop = int(train_ration*length)
    val_stop = int((val_ration + train_ration)*length)
    chunks["train"].append(pd.concat([_true[:train_stop], _false[:train_stop]]))
    chunks["val"].append(pd.concat([(_true[train_stop: val_stop]),
                                     _false[train_stop: val_stop]]))
    chunks["test"].append(pd.concat([_true[val_stop:], _false[val_stop:]]))
    return chunks["train"][0], chunks["val"][0], chunks["test"][0]


def train_model():

    df = pd.read_csv('dataset/fake-and-real-news-dataset/Fake.csv')
    df['Fake'] = 'True'

    df_real = pd.read_csv('dataset/fake-and-real-news-dataset/True.csv')
    df_real['Fake'] = 'False'

    df = pd.concat([df, df_real])

    df = df.sample(len(df))

    df.reset_index()

    df_news = df[['title', 'text', 'Fake']]
    df_news['title_text'] = df['title'] + ' - ' + df['text']

    df_news = df_news[['title_text', 'Fake']]

    # news = np.array(df_news['title_text'])

    # fakes = np.array(df_news['Fake'], dtype='str')
    # len(news) == len(fakes)  # light check

    train, validation, test = split_data(df_news, 0.05, .1)

    train_text = np.array(train['title_text'])
    train_label = np.array(train['Fake'])
    validation_text = np.array(validation['title_text'])
    validation_label = np.array(validation['Fake'])
    test_text = np.array(test['title_text'])
    test_label = np.array(test['Fake'])

    # TOKENIZING
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

    tokenizer.fit_on_texts(train_text)
    # word_index = tokenizer.word_index
    # index_word = tokenizer.index_word
    train_sequences = tokenizer.texts_to_sequences(train_text)
    val_sequences = tokenizer.texts_to_sequences(validation_text)

    # test_sequences = tokenizer.texts_to_sequences(test_text)

    # max_len = max([len(seq) for seq in train_sequences])

    # too long , we can make it 500
    max_len = 500
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_len)
    padded_val_sequences = pad_sequences(val_sequences, maxlen=max_len)
    # padded_test_sequences = pad_sequences(test_sequences, maxlen=max_len)

    # convertintg the labels to one_hot_encodes
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(train_label)
    train_label_sequences = label_tokenizer.texts_to_sequences(train_label)
    validation_label_sequences = \
        label_tokenizer.texts_to_sequences(validation_label)
    test_label_sequences = label_tokenizer.texts_to_sequences(test_label)
    test_label_sequences_ = to_categorical(test_label_sequences)
    validation_label_sequences_ = to_categorical(validation_label_sequences)
    train_label_sequences_ = to_categorical(train_label_sequences)

    # the fist field does not has make any sense ..
    # its onLy True or False from the original dataset
    train_label_sequences_ = train_label_sequences_[:, 1:]
    validation_label_sequences_ = validation_label_sequences_[:, 1:]
    test_label_sequences_ = test_label_sequences_[:, 1:]

    # TRAINING MODEL

    vocab_size = 10000
    # len(word_index) # this variable only represents
    # how many vacalary form the word index we have used
    # to tokenize the sentences , note in our caseO
    # onLy its the same as len(word_index)
    # dimesions = 16
    dimesions = 16
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, dimesions),
        tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    es_ = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
    history = model.fit(padded_train_sequences,
                        train_label_sequences_,
                        epochs=10,
                        batch_size=64,
                        validation_data=(padded_val_sequences,
                                         validation_label_sequences_),
                        callbacks=[es_])

    model.summary()

    model.save('ml_section/resources/trained_models/model.json')
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss, 'b', label='loss')
    plt.plot(val_loss, 'orange', label='val loss')

    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(acc, 'b', label='acc')
    plt.plot(val_acc, 'orange', label='val acc')

    plt.legend()
    plt.show()

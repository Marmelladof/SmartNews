import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# SPLITTING
def split_data(dataframe, train_ration, val_ration):
    _true = dataframe.loc[dataframe.Fake == 0]
    _false = dataframe.loc[dataframe.Fake == 1]

    chunks = {"train": [], "val": [], "test": []}
    length = len(dataframe.index)

    train_stop = int(train_ration*length)
    val_stop = int((val_ration + train_ration)*length)

    chunks["train"].append(pd.concat([_true[:train_stop],
                                      _false[:train_stop]]))
    chunks["val"].append(pd.concat([(_true[train_stop: val_stop]),
                                    _false[train_stop: val_stop]]))
    chunks["test"].append(pd.concat([_true[val_stop:], _false[val_stop:]]))

    return chunks["train"][0], chunks["val"][0], chunks["test"][0]


def train_model():

    df = pd.read_csv('dataset/fake-and-real-news-dataset/Fake.csv')
    df['Fake'] = 1

    df_real = pd.read_csv('dataset/fake-and-real-news-dataset/True.csv')
    df_real['Fake'] = 0

    df = pd.concat([df, df_real])

    df = df.sample(len(df))

    df.reset_index()

    df_news = df[['title', 'text', 'Fake']]
    df_news['title_text'] = df['title'] + ' - ' + df['text']

    df_news = df_news[['title_text', 'Fake']]

    # news = np.array(df_news['title_text'])

    # fakes = np.array(df_news['Fake'], dtype='str')
    # len(news) == len(fakes)  # light check

    train, validation, test = split_data(df_news, 0.4, 0.3)

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
    test_sequences = tokenizer.texts_to_sequences(test_text)

    le = LabelEncoder()
    train_label = le.fit_transform(train_label)
    train_label = to_categorical(train_label, 2)
    validation_label = le.fit_transform(validation_label)
    validation_label = to_categorical(validation_label, 2)
    test_label = le.fit_transform(test_label)
    test_label = to_categorical(test_label, 2)


    # max_len = max([len(seq) for seq in train_sequences])

    # too long , we can make it 500
    # measurer = np.vectorize(len)
    # max_len_test = \
    #     measurer(df_news['title_text'].values.astype(str)).max(axis=0)
    # print('Max article word length:', max_len_test)
    print('Should we use this value as intup? Avoid loosing important info?')
    print('We need to ensure vocabulary robustness for this')
    print('May overcomplicate the problem...')
    max_len = 5000
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_len)
    padded_val_sequences = pad_sequences(val_sequences, maxlen=max_len)
    padded_test_sequences = pad_sequences(test_sequences, maxlen=max_len)

    # # convertintg the labels to one_hot_encodes
    # label_tokenizer = Tokenizer()
    # label_tokenizer.fit_on_texts(train_label)
    # train_label_sequences = label_tokenizer.texts_to_sequences(train_label)
    # validation_label_sequences = \
    #     label_tokenizer.texts_to_sequences(validation_label)
    # test_label_sequences = label_tokenizer.texts_to_sequences(test_label)
    # test_label_sequences_ = to_categorical(test_label_sequences)
    # validation_label_sequences_ = to_categorical(validation_label_sequences)
    # train_label_sequences_ = to_categorical(train_label_sequences)

    # # the fist field does not has make any sense ..
    # # its onLy True or False from the original dataset
    # train_label_sequences_ = train_label_sequences_[:, 1:]
    # validation_label_sequences_ = validation_label_sequences_[:, 1:]
    # test_label_sequences_ = test_label_sequences_[:, 1:]

    # TRAINING MODEL

    vocab_size = 10000
    # len(word_index) # this variable only represents
    # how many vacalary form the word index we have used
    # to tokenize the sentences , note in our caseO
    # onLy its the same as len(word_index)
    # dimesions = 16
    dimesions = 16
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, dimesions))
    model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    es_ = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

    epochs = 10
    history = model.fit(padded_train_sequences,
                        train_label,
                        epochs=epochs,
                        batch_size=64,
                        validation_data=(padded_val_sequences,
                                         validation_label),
                        callbacks=[es_])

    model.summary()

    model.save('ml_section/resources/trained_models/model')
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss, 'b', label='train_loss')
    plt.plot(val_loss, 'orange', label='val_loss')

    plt.legend()
    plt.savefig(f'LOSS_train-{2}_test-{6}_epoch-{epochs}_vocab_size-{vocab_size}.png')  # noqa:E501
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(acc, 'b', label='train_acc')
    plt.plot(val_acc, 'orange', label='val-acc')

    plt.legend()
    plt.savefig(f'LEARNING_train-{2}_test-{6}_epoch-{epochs}_vocab_size-{vocab_size}.png')  # noqa:E501
    plt.show()

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(padded_test_sequences, test_label, batch_size=len(padded_test_sequences))
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(padded_test_sequences[:3])
    print("predictions shape:", predictions.shape)

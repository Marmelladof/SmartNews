import numpy as np
import pandas as pd
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


def data_preprocessing(df, df_real, train_ration=0.4, val_ration=0.3):

    df = pd.concat([df, df_real])

    df = df.sample(len(df))

    df.reset_index()

    df_news = df[['title', 'text', 'Fake']]
    df_news['title_text'] = df['title'] + ' - ' + df['text']

    df_news = df_news[['title_text', 'Fake']]

    # news = np.array(df_news['title_text'])

    # fakes = np.array(df_news['Fake'], dtype='str')
    # len(news) == len(fakes)  # light check

    train, validation, test = split_data(df_news, train_ration, val_ration)

    train_text = np.array(train['title_text'])
    train_label = np.array(train['Fake'])
    validation_text = np.array(validation['title_text'])
    validation_label = np.array(validation['Fake'])
    test_text = np.array(test['title_text'])
    test_label = np.array(test['Fake'])

    # TOKENIZING
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

    tokenizer.fit_on_texts(train_text)
    word_index = tokenizer.word_index
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

    '''
        Should we use this value as intup? Avoid loosing important info?
        We need to ensure vocabulary robustness for this
        May overcomplicate the problem...
    max_len = max([len(seq) for seq in train_sequences])
    too long , we can make it 500
    measurer = np.vectorize(len)
    max_len_test = \\
        measurer(df_news['title_text'].values.astype(str)).max(axis=0)
    print('Max article word length:', max_len_test)
    '''

    max_len = 1000
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_len)
    padded_val_sequences = pad_sequences(val_sequences, maxlen=max_len)
    padded_test_sequences = pad_sequences(test_sequences, maxlen=max_len)

    return (padded_train_sequences,
            padded_val_sequences,
            padded_test_sequences,
            train_label,
            validation_label,
            test_label,
            word_index)


def preprocessing(train_ration=None, val_ration=None):

    df = pd.read_csv('dataset/fake-and-real-news-dataset/Fake.csv')
    df['Fake'] = 1

    df_real = pd.read_csv('dataset/fake-and-real-news-dataset/True.csv')
    df_real['Fake'] = 0

    padded_train_sequences, padded_val_sequences, padded_test_sequences, train_label, validation_label, test_label, word_index = data_preprocessing(df, df_real, train_ration, val_ration)  # noqa:E501

    return (padded_train_sequences,
            padded_val_sequences,
            padded_test_sequences,
            train_label,
            validation_label,
            test_label,
            word_index)


def preprocessing_for_test():

    df = pd.read_csv('dataset/fake-and-real-news-dataset/Fake.csv')
    df['Fake'] = 1

    df_real = pd.read_csv('dataset/fake-and-real-news-dataset/True.csv')
    df_real['Fake'] = 0

    padded_train_sequences, padded_val_sequences, padded_test_sequences, train_label, validation_label, test_label, word_index = data_preprocessing(df, df_real)  # noqa:E501

    return (padded_test_sequences, test_label)

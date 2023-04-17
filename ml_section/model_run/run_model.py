from tensorflow import keras
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# TOKENIZING
def tokenizer(raw_input_data):
    input_data = \
        np.array([str(raw_input_data["Title"] + raw_input_data["Text"])])
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

    tokenizer.fit_on_texts(input_data)
    input = tokenizer.texts_to_sequences(input_data)
    max_len = 500
    input_data_sequences = pad_sequences(input, maxlen=max_len)

    return input_data_sequences


def load_model():
    model = keras.models.load_model('ml_section/resources/trained_models/model')
    return model


def run_model(raw_input_data):
    model = load_model()
    model.predict(x=raw_input_data)


if __name__ == '__main__':
    raw_input = {
        "Title": "New Martian technology discovered",
        "Text": "After the Pope's last visit to the Andromeda Galaxy he took the oportunity to stop by Mars. Here he preached his new anti-abortion tactics and how he had managed to make homossecuality a crime in Europe, once more."
    }
    input_data = tokenizer(raw_input)
    run_model(input_data)

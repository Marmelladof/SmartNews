from tensorflow import keras
import numpy as np
# from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences


model = \
        keras.models.load_model('ml_section/resources/trained_models/model')


# TOKENIZING
def tokenizer(raw_input_data):
    input_data = \
        np.array([str(raw_input_data["Title"] + raw_input_data["Text"])])

    return input_data


# def load_model():
#     model = \
#         keras.models.load_model('ml_section/resources/trained_models/model')
#     tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

#     tokenizer.fit_on_texts(input_data)
#     input = tokenizer.texts_to_sequences(input_data)
#     max_len = 500
#     input_data_sequences = pad_sequences(input, maxlen=max_len)
#     return model


def run_model(raw_input):
    input_data = tokenizer(raw_input)
    prediction = model.predict(x=input_data)
    if prediction > 0.5:
        label = 'Fake'
    else:
        label = 'Real'
    return prediction, label


if __name__ == '__main__':
    raw_input = {
        "Title": "New Martian technology discovered",
        "Text": "After the Pope's last visit to the Andromeda Galaxy he took "
        "the oportunity to stop by Mars. Here he preached his new "
        "anti-abortion tactics and how he had managed to make homossexuality "
        "a crime in Europe, once more. He is now making advances in driving "
        "kids into shooting sprees in GOD's name all across the world. The "
        "new cutting edge technolagy shared by the Martians allows them to "
        "summon black holes wherever they wish and they got the world "
        "trembling in fear of them."
    }
    prediction, label = run_model(raw_input)
    print(f'The NEWS article provided is considered {label} with a certanty of {prediction}')  # noqa: E501

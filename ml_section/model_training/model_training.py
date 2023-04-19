import tensorflow as tf
# import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from ml_section.model_training.pre_processing import preprocessing
from ml_section.model_testing.model_testing import test_model_trained
from keras import layers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus, 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


def train_model():

    # PRE-PROCESS DATA
    padded_train_sequences, padded_val_sequences, padded_test_sequences, train_label, validation_label, test_label, word_index = preprocessing(train_ration=0.3, val_ration=0.3)  # noqa: E501
    vocab_size = 10000
    # vocab_size = len(word_index)
    with open("word_index.json", "w") as file:
        import json
        # Serializing json
        json_object = json.dumps(word_index, indent=4)
        file.write(json_object)
    print(word_index)
    # len(word_index) # this variable only represents
    # how many vacalary form the word index we have used
    # to tokenize the sentences

    # model.add(tf.keras.layers.Bidirectional(
    #             tf.keras.layers.LSTM(64, return_sequences=True)))
    # model.add(tf.keras.layers.Bidirectional(
    #             tf.keras.layers.LSTM(32, return_sequences=True)))
    # model.add(tf.keras.layers.Bidirectional(
    #             tf.keras.layers.LSTM(64, return_sequences=True)))

    # Bidirectional LSTM
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

    # 1D Convolutional Layers + 1D Max Pooling
    # features = 20000
    # seq_len = 500

    # vectorization_layer = layers.TextVectorization(
    #     max_tokens=features,
    #     output_mode='int',
    #     output_sequence_length=seq_len,
    #     standardize='lower_and_strip_punctuation')
    # vectorization_layer.adapt(padded_train_sequences)

    # txt_input = tf.keras.Input(shape=(1,), dtype=tf.string)
    # x = vectorization_layer(txt_input)
    # x = layers.Embedding(features+1, 128)(x)
    # x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)  # noqa: E501
    # x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)  # noqa: E501
    # x = layers.GlobalMaxPooling1D()(x)
    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.5)(x)
    # output = layers.Dense(1, activation="sigmoid")(x)

    # model = tf.keras.Model(txt_input, output)
    # model.compile(loss="binary_crossentropy",
    #               optimizer="adam",
    #               metrics=["accuracy"])

    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
    model.summary()

    epochs = 5
    # TRAINING MODEL
    history = model.fit(padded_train_sequences,
                        train_label,
                        epochs=epochs,
                        batch_size=64,
                        validation_data=(padded_val_sequences,
                                         validation_label),
                        callbacks=[callback])

    # history = model.fit(
    #             x=padded_train_sequences,
    #             y=train_label,
    #             epochs=epochs,
    #             validation_data=(padded_val_sequences,
    #                              validation_label),
    #             shuffle=True,
    #             callbacks=[callback])

    model.save('ml_section/resources/trained_models/model_final')
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(loss, 'b', label='train_loss')
    plt.plot(val_loss, 'orange', label='val_loss')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(acc, 'b', label='train_acc')
    plt.plot(val_acc, 'orange', label='val-acc')

    plt.legend()
    plt.savefig('LEARNING_LOSS_image.png')  # noqa:E501
    plt.show()

    # TEST MODEL
    test_model_trained(model, padded_test_sequences, test_label)

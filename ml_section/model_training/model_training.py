import tensorflow as tf
# import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from ml_section.model_training.pre_processing import preprocessing
from ml_section.model_testing.model_testing import test_model_trained

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=12288  # set your limit
                    )
                ],
            )
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


def train_model():

    # PRE-PROCESS DATA
    padded_train_sequences, padded_val_sequences, padded_test_sequences, train_label, validation_label, test_label, word_index = preprocessing(train_ration=0.1, val_ration=0.1)  # noqa: E501
    vocab_size = 10000
    print(word_index)
    # len(word_index) # this variable only represents
    # how many vacalary form the word index we have used
    # to tokenize the sentences

    dimesions = 16
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, dimesions))
    model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)))
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
    # TRAINING MODEL
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

    # TEST MODEL
    test_model_trained(model, padded_test_sequences, test_label)

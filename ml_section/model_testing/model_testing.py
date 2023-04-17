from tensorflow import keras
from ml_section.model_training.pre_processing import preprocessing_for_test


def test_model_trained(model, padded_test_sequences, test_label):
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(padded_test_sequences,
                             test_label,
                             batch_size=len(padded_test_sequences))
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(padded_test_sequences[:3])
    print("predictions shape:", predictions.shape)


def test_model_saved():

    model = \
        keras.models.load_model('ml_section/resources/trained_models/model')

    # PRE-PROCESS DATA
    padded_test_sequences, test_label = preprocessing_for_test()  # noqa: E501

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(padded_test_sequences,
                             test_label,
                             batch_size=len(padded_test_sequences))
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(padded_test_sequences[:3])
    print("predictions shape:", predictions.shape)

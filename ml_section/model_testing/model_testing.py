from tensorflow import keras
from ml_section.model_training.pre_processing import preprocessing_for_test

from sklearn.metrics import confusion_matrix as conf_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

import matplotlib.pyplot as plt


def test_model_trained(model, padded_test_sequences, test_label):
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(padded_test_sequences,
                             test_label,
                             batch_size=len(padded_test_sequences))
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions")
    predictions = model.predict(padded_test_sequences)
    predictions = [1 if x[0] > 0.5 else 0 for x in predictions]
    # print(predictions)
    # predictions = [1 if x[0] > 0.5 else 0 for x in predictions]
    # print(predictions)
    # test_label = [x[0] for x in test_label]
    _conf_matrix = conf_matrix(test_label, predictions)
    _accuracy_score = accuracy_score(predictions, test_label)
    print(_accuracy_score)

    fig1 = plt.figure(figsize=(8, 8))
    sns.heatmap(_conf_matrix, annot=True)
    plt.xlabel("Predictions")
    plt.ylabel("Actuals")
    plt.title('Confusion Matrix', fontsize=18)
    fig1.savefig("confusion_matrix_model.png")


def test_model_saved():

    model = \
        keras.models.load_model('ml_section/resources/trained_models/model')

    # PRE-PROCESS DATA
    padded_test_sequences, test_label = preprocessing_for_test()  # noqa: E501
    print(padded_test_sequences)
    print(test_label)
    test_label = [x[0] for x in test_label]
    print(test_label)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    # results = model.evaluate(padded_test_sequences,
    #                          test_label,
    #                          batch_size=len(padded_test_sequences))
    # print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(padded_test_sequences)
    print(predictions)
    predictions = [x[0] for x in predictions]
    predictions = [1 if x > 0.5 else 0 for x in predictions]
    print(predictions)
    _conf_matrix = conf_matrix(test_label, predictions)
    _accuracy_score = accuracy_score(predictions, test_label)
    print(_accuracy_score)

    fig1 = plt.figure(figsize=(8, 8))
    sns.heatmap(_conf_matrix, annot=True)
    plt.xlabel("Predictions")
    plt.ylabel("Actuals")
    plt.title('Confusion Matrix', fontsize=18)
    fig1.savefig("confusion_matrix_model.png")


if __name__ == '__main__':
    test_model_saved()

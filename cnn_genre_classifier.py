import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# DATA_SET_PATH
DATA_PATH = "kpopdata.json"


def load_data(data_path, max_sequence_length):
    """Loads training dataset from json file.

    :param data_path (str): Path to json file containing data
    :param max_sequence_length (int): Maximum sequence length for MFCC features
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    tracks = data["tracks"]

    X_mfcc = [track["mfcc"] for track in tracks]
    X_mfcc_padded = keras.preprocessing.sequence.pad_sequences(X_mfcc, maxlen=max_sequence_length, padding='post',
                                                               dtype='float32')

    X_other_features = np.array([
        [track["danceability"], track["energy"], track["loudness"], track["tempo"], track["valence"]]
        for track in tracks
    ])

    X_other_features_reshaped = np.repeat(X_other_features[:, np.newaxis, :], max_sequence_length, axis=1)

    X = np.concatenate((X_mfcc_padded, X_other_features_reshaped), axis=2)
    y = np.array([track["label"] for track in tracks])

    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

    :param history: Training history of the model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size, max_sequence_length):
    X, y = load_data(DATA_PATH, max_sequence_length)

    # Flatten the 3D MFCC sequences to 2D
    X_flattened = X.reshape(X.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=test_size, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size,
                                                                    random_state=42)

    # Standardize the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    # Save the scaler mean and scale
    scaler_mean_path = 'scaler_mean.npy'
    scaler_scale_path = 'scaler_scale.npy'

    np.save(scaler_mean_path, scaler.mean_)
    np.save(scaler_scale_path, scaler.scale_)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, num_classes):
    model = keras.Sequential()

    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())  # Flatten the input

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return model


def train_model(X_train, y_train, X_validation, y_validation, input_shape, num_classes, epochs=30):
    model = build_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_validation, y_validation),
        batch_size=32,
        epochs=epochs
    )

    return model, history





def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # perform prediction
    prediction = model.predict(X.reshape(1, -1))

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


if __name__ == "__main__":
    max_sequence_length = 100  # Adjust this value based on your dataset
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2, max_sequence_length)
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))
    model, history = train_model(X_train, y_train, X_validation, y_validation, input_shape, num_classes=2, epochs=50)

    plot_history(history)

    model.save("model.h5")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Pick a sample to predict from the test set
    sample_idx = 100
    X_to_predict = X_test[sample_idx]
    y_to_predict = y_test[sample_idx]

    # Predict sample
    predict(model, X_to_predict, y_to_predict)

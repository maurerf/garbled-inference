from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np


def replace_af(original_model, activation_function):
    """
    Replaces all activation functions of a model.

    :param original_model: The model to be modifed. Be wary of state! This change is permanent.
    :param activation_function: Function mapping (28,28,1) -> (28,28,1).
    """
    for layer in original_model.layers:
        if hasattr(layer, "activation"):
            layer.activation = activation_function
    original_model.compile(loss=keras.losses.mean_squared_error, optimizer="Adam", metrics=["accuracy"])

    # test accuracy without retraining
    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("pre training accuracy %.3f" % (acc * 100.0))

    # retrain
    original_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # test accuracy after fitting
    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("post training accuracy: %.3f" % (acc * 100.0))


# todo: these approximations dont operate on scalars but on (28, 28, 1)
def relu_approx_1(x):
    """
    Square approximation
    """
    return x**2


def relu_approx_2(x):
    """
    Deg. 2 approximation:
    0.375373 + 0.5x + 0.117071x²
    source: https://arxiv.org/pdf/2009.03727.pdf, table 2
    """
    return 0.375373 + (0.5*x) + (0.117071 * (x**2))


def relu_approx_3(x):
    """
    Deg. 4 approximation:
    0.234606 + 0.5x + 0.204875x²− 0.0063896x⁴
    source: https://arxiv.org/pdf/2009.03727.pdf, table 2
    """
    return 0.234606 + (0.5 * x) + (0.204875 * (x**2)) - (0.0063896 * (x**4))


def sign_approx_1(x):
    return x  # todo: find approximations for sign


def sign_approx_2(x):
    return x  # todo: find approximations for sign


def sign_approx_3(x):
    return x  # todo: find approximations for sign


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    x_train = x_train.astype("float64") / 255
    x_test = x_test.astype("float64") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    batch_size = 128
    epochs = 1

    # define original model
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(8, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)),
            layers.Flatten(),
            layers.Dense(10, activation=None),
        ]
    )

    # model.summary()

    # compile original model
    model.compile(loss=keras.losses.mean_squared_error, optimizer="Adam", metrics=["accuracy"])

    # train original model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # test accuracy of original
    _, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\n\n--- ORIGINAL ---")
    print("accuracy:  %.3f" % (acc * 100.0))

    # use approximations of relu
    # todo: clean up this model copying mess
    model_copy = model
    print("\n\n--- RELU APPROX 1 ---")
    replace_af(model_copy, relu_approx_1)
    model_copy = model
    print("\n\n--- RELU APPROX 2 ---")
    replace_af(model_copy, relu_approx_2)
    model_copy = model
    print("\n\n--- RELU APPROX 3 ---")
    replace_af(model_copy, relu_approx_3)
    model_copy = model

    # use approximations of sign
    # print("\n\n--- SIGN APPROX 1 ---")
    # replace_af(model, sign_approx_1)
    # print("\n\n--- SIGN APPROX 2 ---")
    # replace_af(model, sign_approx_2)
    # print("\n\n--- SIGN APPROX 3 ---")
    # replace_af(model, sign_approx_3)

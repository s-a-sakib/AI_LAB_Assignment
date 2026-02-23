import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# ----------------------------
# Build FCFNN Model
# ----------------------------
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)

    x = Dense(8, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ----------------------------
# Training Function
# ----------------------------
def train_and_evaluate(dataset_name, load_function):
    print(f"\n========== {dataset_name} ==========")

    (x_train, y_train), (x_test, y_test) = load_function()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = build_model(x_train.shape[1:], num_classes)

    model.summary()

    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"{dataset_name} Test Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} Test Loss: {loss:.4f}")


# ----------------------------
# Run All Three Datasets
# ----------------------------

train_and_evaluate("Fashion MNIST", tf.keras.datasets.fashion_mnist.load_data)
train_and_evaluate("MNIST", tf.keras.datasets.mnist.load_data)
train_and_evaluate("CIFAR-10", tf.keras.datasets.cifar10.load_data)
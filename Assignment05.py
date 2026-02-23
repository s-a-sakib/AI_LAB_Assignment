from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np

def build_cnn(num_classes, input_shape=(28,28,1)):
    inputs = Input(input_shape)
    x = Conv2D(8, (3,3), activation='relu', name='conv_layer_1')(inputs)
    x = Conv2D(16, (3,3), activation='relu', name='conv_layer_2')(x)
    x = Flatten(name='flatten_layer')(x)
    x = Dense(128, activation='relu', name='hidden_layer_1')(x)
    x = Dense(16, activation='relu', name='hidden_layer_2')(x)
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)
    model = Model(inputs, outputs)
    model.summary()
    return model

def train_and_evaluate(dataset_name, load_function):
    print(f"\n========== {dataset_name} ==========")

    (x_train, y_train), (x_test, y_test) = load_function()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape for Conv2D
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Determine number of classes
    num_classes = len(np.unique(y_train))

    # Build model
    model = build_cnn(num_classes)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_and_evaluate("MNIST", mnist.load_data)
    train_and_evaluate("Fashion MNIST", fashion_mnist.load_data)
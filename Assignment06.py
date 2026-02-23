# Write a report in pdf format using any Latex system after:
#       ● Prepare an English handwritten digit dataset by collecting hand written data and
#         splitting into the training set and test.

#       ● Retrain FCFNN using your training set with the training set of the MNIST English
#         digit dataset.
#       ● Evaluate your FCFNN using your test set along with the test set of the MNIST
#         English dataset.
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os


# -------------------------------------------------
# 1️⃣ Build FCFNN Model
# -------------------------------------------------
def build_fcfnn(input_shape, num_classes):

    inputs = Input(shape=input_shape)

    x = Flatten()(inputs)  # IMPORTANT
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


# -------------------------------------------------
# 2️⃣ Load MNIST
# -------------------------------------------------
def load_mnist_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # reshape for model
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


# -------------------------------------------------
# 3️⃣ Load Custom Dataset (YOUR FOLDER)
# -------------------------------------------------
def load_custom_data():

    directory = "2110976109"
    csv_path = os.path.join(directory, "labels.csv")

    df = pd.read_csv(csv_path)

    images = []
    labels = []

    for index, row in df.iterrows():
        img_path = os.path.join(directory, row["filename"])

        img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))
        img_array = image.img_to_array(img)

        images.append(img_array)
        labels.append(row["label"])

    images = np.array(images).astype("float32") / 255.0
    labels = np.array(labels)

    # Split custom dataset (80/20)
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    return (x_train, y_train), (x_test, y_test)


# -------------------------------------------------
# 4️⃣ MAIN
# -------------------------------------------------
if __name__ == "__main__":

    # Load MNIST
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_mnist_data()

    # Load Custom
    (x_train_custom, y_train_custom), (x_test_custom, y_test_custom) = load_custom_data()

    # Combine Training Sets
    x_train = np.concatenate((x_train_mnist, x_train_custom), axis=0)
    y_train = np.concatenate((y_train_mnist, y_train_custom), axis=0)

    num_classes = 10

    model = build_fcfnn(input_shape=(28, 28, 1), num_classes=num_classes)

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

    # Evaluate on MNIST Test
    print("\nEvaluation on MNIST Test Set:")
    loss_mnist, acc_mnist = model.evaluate(x_test_mnist, y_test_mnist, verbose=0)

    # Evaluate on Custom Test
    print("\nEvaluation on Custom Test Set:")
    loss_custom, acc_custom = model.evaluate(x_test_custom, y_test_custom, verbose=0)

    print(f"\nMNIST Accuracy: {acc_mnist:.4f}")
    print(f"Custom Dataset Accuracy: {acc_custom:.4f}")
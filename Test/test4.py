import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os

def model_cnn(num_classes, shape = (28,28,1)):
    inputs = Input(shape, name = "input_layer")
    x = Conv2D(16, (3,3), activation = "relu")(inputs)
    x = Conv2D(8, (3,3), activation = "relu")(x)
    
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten(name = "flatten_layer")(x)
    x = Dense(16, activation = "relu")(x)
    x = Dense(32, activation = "relu")(x)
    outputs = Dense(num_classes, activation = "softmax", name = "output_layer")(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary(show_trainable = True)
    return model

filepath = "mnist.npz"
data = np.load(filepath)

x_train = data["x_train"]
y_train = data["y_train"]

x_data = data["x_test"]
y_data = data["y_test"]

length = len(x_data)

t = int(length * .5)

x_test = x_data[0:t]
y_test = y_data[0:t]

x_val = x_data[t: length]
y_val = y_data[t: length]


num_classes = len(np.unique(y_test))
x_train = x_train / 255.0
x_test  = x_test  / 255.0
x_val   = x_val  / 255.0

model = model_cnn(10)
model.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train,y_train, validation_data=(x_val,y_val), epochs = 1)

loss , accuracy = loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

plt.figure(figsize=(12,8))
plt.plot(history.history["accurecy"], label = "Accuracy", color = "green")
plt.plot(history.history["val_accuracy"], label = "Val Accuracy", color = "red")
plt.title("Accuracy")
plt.xlabel("accuracy")
plt.ylabel("Epoch")
plt.grid(True)
plt.show()


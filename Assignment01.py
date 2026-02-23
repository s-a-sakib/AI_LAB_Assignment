import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Assignment 1
inputs = Input(shape=(8,), name="input layer")
h1 = Dense(4, activation="relu")(inputs)
h2 = Dense(8, activation="relu")(h1)
h3 = Dense(4, activation="relu")(h2)
outputs = Dense(10, activation="sigmoid")(h3)

model = Model(inputs=inputs, outputs=outputs)
model.summary(show_trainable=True)

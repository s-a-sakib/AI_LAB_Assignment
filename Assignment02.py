import tensorflow as ts
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(8,), name = "Input Layer")
h1 = Dense(4, activation="relu")(inputs)
h2 = Dense(8, activation="relu")(h1)
h3 = Dense(4, activation="relu")(h2)
outputs = Dense(10, activation="sigmoid")(h3)

model = Model(input = inputs, output= outputs)
model.summary(show_trainable=True)
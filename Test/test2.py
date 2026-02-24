import tensorflow as tf
from tensorflow.keras.layers import Input,Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

    
def model(x,y):
    inputs = Input(shape = (x,y),name = "input layer")
    x = Flatten(name = "Flatten_Layer")(inputs)
    h1 = Dense(16, activation = "relu")(x)
    h2 = Dense(32, activation = "relu")(h1)
    h3 = Dense(64, activation = "relu")(h2)
    h4 = Dense(32, activation = "relu")(h3)
    outputs = Dense(10, activation = "softmax")(h4)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary(show_trainable = True)
    return model

model = model(28,28)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test  = x_test / 255.0

model.fit(x_train,y_train, validation_data =(x_test,y_test), epochs = 50)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

index = np.random.randint(0,50,10)
images = x_test[index]
valuse = y_test[index]

predictions = model.predict(images)

for i in range(10):
    plt.imshow(images[i], cmap="gray")
    plt.title(f"True: {valuse[i]}  |  Pred: {np.argmax(predictions[i])}")
    plt.axis("off")
    plt.show()


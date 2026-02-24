import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

def model_cnn(numOfOutput, shape=(28,28,1)):
    inputs = Input(shape, name="input_layer")

    x = Conv2D(8, (3,3), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(16,(3,3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(numOfOutput, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.summary()
    return model
    
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test  / 255.0

num_classes = len(np.unique(y_test))

model = model_cnn(num_classes)
model.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 5)

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


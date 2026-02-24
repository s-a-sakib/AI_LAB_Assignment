import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

def model(inputshape):
    inputs = Input(shape = (inputshape,), name = "input layer")
    h1 = Dense(16, activation = "relu")(inputs)
    h2 = Dense(32, activation = "relu")(h1)
    h3 = Dense(64, activation = "relu")(h2)
    outputs = Dense(1, activation = "linear")(h3)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary(show_trainable = True)
    return model
    
def equation1(x):
    return 5 * x + 10
    
def equation2(x):
    return 3 * x * x + 5 * x + 10

x_train = np.random.uniform(-10,10,10000)
x_val   = np.random.uniform(-10,10,400)
x_test  = np.random.uniform(-10,10,200)

y_train = equation2(x_train)
y_val   = equation2(x_val)
y_test  = equation2(x_test)

x_train = x_train.reshape(-1, 1)
x_val   = x_val.reshape(-1, 1)
x_test  = x_test.reshape(-1, 1)

y_train = y_train.reshape(-1, 1)
y_val   = y_val.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)


model = model(1)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32
)

test_loss, test_mae = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test MAE:", test_mae)

predictions = model.predict(x_test)
plt.scatter(x_test, y_test, label='Original y')
plt.scatter(x_test, predictions, label='Predicted y')
plt.title("Original vs Predicted (y = 3 * x * x + 5 * x + 10)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

y = equation2(10)
y1 = model.predict(np.array([[10]]))

print(y)
print(y1)

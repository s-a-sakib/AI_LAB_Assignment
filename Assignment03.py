# Write a report in pdf format using any Latex system after:
# ● building FCFNNs for solving the following equations:
# i. y = 5x + 10
# ii. y = 3x^2 + 5x + 10

# iii. y = 4x^3 + 3x^2 + 5x + 10

# ● preparing a training set, a validation set and a test set for the above equations.
# ● training and testing FCFNNs using your prepared data.
# ● plotting original y and ‘predicted y’.
# ● explaining the effect of "power of an independent variable" on the architecture of
# your FCFNN and the amount of training data.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# -------------------------------
# 1. Define Equation
# -------------------------------
def equation1(x):
    return 5 * x + 10

def equation2(x):
    return 3 * x**2 + 5 * x + 10

def equation3(x):
    return 4 * x**3 + 3 * x**2 + 5 * x + 10
# -------------------------------
# 2. Prepare Dataset
# -------------------------------
# Generate random x values
x_train = np.random.uniform(-10, 10, 1000)
x_val   = np.random.uniform(-10, 10, 200)
x_test  = np.random.uniform(-10, 10, 200)

# Compute y values
y_train = equation1(x_train)
y_val   = equation1(x_val)
y_test  = equation1(x_test)

# Reshape for TensorFlow (IMPORTANT)
x_train = x_train.reshape(-1, 1)
x_val   = x_val.reshape(-1, 1)
x_test  = x_test.reshape(-1, 1)

y_train = y_train.reshape(-1, 1)
y_val   = y_val.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)

# -------------------------------
# 3. Build FCFNN Model
# -------------------------------
inputs = Input(shape=(1,))
h1 = Dense(16, activation='relu')(inputs)
h2 = Dense(16, activation='relu')(h1)
h3 = Dense(16, activation='relu')(h2)
outputs = Dense(1, activation='linear')(h3)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# -------------------------------
# 4. Train Model
# -------------------------------
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32
)

# -------------------------------
# 5. Test Model
# -------------------------------
test_loss, test_mae = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test MAE:", test_mae)

# -------------------------------
# 6. Predict
# -------------------------------
predictions = model.predict(x_test)

# -------------------------------
# 7. Accrecy
# -------------------------------
accuracy = 1 - np.mean(np.abs(predictions - y_test) / np.abs(y_test))
print(f"Accuracy: {accuracy:.4f}")

# -------------------------------
# 8. Plot Original vs Predicted
# -------------------------------
plt.scatter(x_test, y_test, label='Original y')
plt.scatter(x_test, predictions, label='Predicted y')
plt.title("Original vs Predicted (y = 4x^3 + 3x^2 + 5x + 10)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
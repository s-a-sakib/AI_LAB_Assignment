from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np

def build_vgg16_style(num_classes, input_shape=(28,28,1)):
    inputs = Input(input_shape, name='input_layer')

    # Block 1
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2,2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2,2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = MaxPooling2D((2,2), name='block3_pool')(x)

    # Fully Connected
    x = Flatten(name='flatten_layer')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)

    model = Model(inputs, outputs)
    model.summary()
    return model

def train_dataset(dataset_name, load_func):
    print(f"\n========== {dataset_name} ==========")
    (x_train, y_train), (x_test, y_test) = load_func()
    x_train = x_train.reshape(-1,28,28,1)/255.0
    x_test  = x_test.reshape(-1,28,28,1)/255.0
    num_classes = len(np.unique(y_train))
    
    model = build_vgg16_style(num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
    
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_dataset("MNIST", mnist.load_data)
    train_dataset("Fashion MNIST", fashion_mnist.load_data)
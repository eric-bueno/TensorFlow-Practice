import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
# don't train using all of the data, reserve some for testing

# normally we need to load in data and manipulate it; keras makes this easier
(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = train_images / 255.0

# model has input, one middle, and one output layer
# input layer shape is 28x28 pixel array (728 nodes)
# middle uses relu (rectify linear unit). a fast, versatile activation function
# output layer has 10 nodes representing each class name
#   softmax ensures outputs sum up to 1 so we can assign probabilities
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the data
# epochs decide how many times the model will see the data
#   randomly pick images and labels to feed to neural network
#   random because order of the input will affect the tweaks made by the model
model.fit(train_images, train_labels, epochs=5)

# use the model
prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + class_names[test_labels[i]])
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])
    plt.show()

# print(class_names[np.argmax(prediction[0])]) #finds largest value (what the model predicted)
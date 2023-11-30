import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

datasets = tf.keras.datasets

(train_images, train_labels), (test_images, test_labels) =\
    datasets.cifar10.load_data()
# Normalize the data we end up the values in between 0 and 1.
train_images, test_images = train_images/255.0, test_images/255.0

class_names = [
    'Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship',
    'Truck',
]

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])

plt.show()

models = tf.keras.models
layers = tf.keras.layers

"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)

test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
model.save('image_and_object_recognition.model')
"""

model = tf.keras.models.load_model('image_and_object_recognition.model')

img1 = cv.imread('car.jpg')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

img2 = cv.imread('horse.jpg')
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

img3 = cv.imread('cat.jpg')
img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)

prediction = model.predict(np.array([img1]) / 255)
index = np.argmax(prediction)
label1 = class_names[index]

prediction2 = model.predict(np.array([img2]) / 255)
index2 = np.argmax(prediction2)
label2 = class_names[index2]

prediction3 = model.predict(np.array([img3]) / 255)
index3 = np.argmax(prediction3)
label3 = class_names[index3]

plt.imshow(img1, cmap=plt.cm.binary)
plt.title('Predict Value: ' + label1)
plt.show()

plt.imshow(img2, cmap=plt.cm.binary)
plt.title('Predict Value: ' + label2)
plt.show()

plt.imshow(img3, cmap=plt.cm.binary)
plt.title('Predict Value: ' + label3)
plt.show()

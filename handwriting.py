import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()

#start layer training

if os.path.isdir("handwritten.model"):

    image_number = 1
    while os.path.isfile(f"numbers/number{image_number}.png"):
            try:
                img = cv2.imread(f"numbers/number{image_number}.png")[:,:,0]
                img = np.invert(np.array([img]))
                pro = model.predict(img)
                print(f"This is hopefully a {np.argmax(pro)}")
                plt.imshow(img[0], cmap=plt.cm.binary)
                plt.show()
            except: 
                print("Unable to recognize")
            finally: 
                image_number += 1
else:

    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax ))

    model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=4)

    model.save('handwritten.model')

    model = tf.keras.models.load_model('handwritten.model')

image_number = 1
while os.path.isfile(f"numbers/number{image_number}.png"):
    try:
        img = cv2.imread(f"numbers/number{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        pro = model.predict(img)
        print(f"This is hopefully a {np.argmax(pro)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except: 
        print("Unable to recognize")
    finally: 
        image_number += 1

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)


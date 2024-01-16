import os
import numpy as np #working with arrays
import cv2  #open-cvm for computer vision
import matplotlib.pyplot as plt
import tensorflow as tf #for machine learning

#this loads the dataset from tensor flow
mnist = tf.keras.datasets.mnist

#now split the data into training data and testing data -- the mnist data set is already split up for us. 
#x is pixel data, y is category
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#now normalize -- so every value between 0 and 1 ex. gray scale
X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)

#create model
#Sequential is a neural network with multiple lauers
model = tf.keras.models.Sequential()
#add layer
#Flatten layer means that we flatten a certain input shape. no longer a grid of 28 by 28 , turns into one big line of pixels
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#now a dense layer, basic neural network layer. each neuron is connected to the next neuron 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #128 is how many units
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) #each unit will represent each individual unit
#softmax makes sures that the output of all 10 units add up to 1
#at the end each unit will have a value between 0 and 1, which will signal how likely the image is that digit

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'])
#run epochs, how many times it will train)
model.fit(X_train, y_train, epochs = 3)
model.save('handwritten.model')

loss, accuracy = model.evaluate(X_test, y_test)
print(loss)
print(accuracy)

model = tf.keras.models.load_model('handwritten.model')

# Load custom images and predict them
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceekj ding with next image...")
        image_number += 1
     
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist= tf.keras.datasets.mnist # 28 x 28 images of handwritten digit  b/w 0-9
                                                            
(x_train, y_train),(x_test,y_test) = mnist.load_data()    #load data set in x,y

# Normalise features it makes it easier NN to predict
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)


'''
Model Architecture:
Input Layer --> Flatten Layer (1 -D array of pixel of images of 28x28 = 784)
No. of hidden layers --> 2 (Dense)
Activation function --> relu (Rectified Linear)
Output Layer --> Dense layer
Activation function -->softmax (for probability distribution)
'''

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation =tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation =tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation =tf.nn.softmax))

# model created
# Run the Model
# Optimizer --> AdamOptimizer

model.compile(optimizer='adam' ,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
# run model for test data

val_loss ,val_acc =model.evaluate(x_test,y_test)
print(val_loss,val_acc)

#model.save('epic_number_reader.model')

#new_model= tf.keras.models.load_model('epic_number_reader.model')

prediction=model.predict(x_test)
print(prediction)
# we get output again as probability distribution.
print(np.argmax(prediction[0]))

plt.imshow(x_test[0])
plt.show()


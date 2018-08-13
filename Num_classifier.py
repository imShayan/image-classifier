import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist= tf.keras.datasets.mnist # 28 x 28 images of handwritten digit  b/w 0-9
#load data set in x,y
(x_train, y_train),(x_test,y_test) = mnist.load_data()

#plt.imshow(x_train[0])
#plt.show()
#print(x_train[0])
#Normalise features it makes it easier NN to predict
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)
#print(x_train[0])
#design the model :
'''
input layer -> flatten array of pixel of images
2 hideen layer ->dense
activation function->relu
output layer ->dense avtivatiojn function ->softmax ->for probability distribution
'''

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation =tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation =tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation =tf.nn.softmax))
# model created

model.compile(optimizer='adam' ,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)

val_loss ,val_acc =model.evaluate(x_test,y_test)
print(val_loss,val_acc)
#model.save('epic_number_reader.model')

#new_model= tf.keras.models.load_model('epic_number_reader.model')
prediction=model.predict(x_test)
print(prediction)

print(np.argmax(prediction[0]))

plt.imshow(x_test[0])
plt.show()


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten

# 1. Load and preprocess the dataset
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

# normalizing the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Define a simple neural network model for classification


model= Sequential()
model.add(Flatten(input_shape=x_train[0].shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(11,activation='softmax'))
model.summary()

# 3. compile the model

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# 4. Train the model
history = model.fit(x_train,y_train,epochs=20,validation_split=0.2)

# 5. Evaluate the model and plot performance indicators
# 6. Plot training and validation loss over epochs

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# 7. Plot training and validation accuracy over epochs

y_prob = model.predict(x_test)
y_pred = y_prob.argmax(axis=1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# 8. Display test loss and accuracy

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# sources: https://www.tensorflow.org/datasets/keras_example
# https://medium.com/@azimkhan8018/a-beginners-guide-to-deep-learning-with-mnist-dataset-0894f7183344
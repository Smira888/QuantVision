#library import

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile

#extracted zip for dataset

zip_path = 'dataset.zip'
extract_path = 'dataset/'
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

#data preparation 

trainpath = 'dataset/Train'
testpath = 'dataset/Test'

imgsize = 160
batchsize = 32
epoch = 25

traindata = ImageDataGenerator(rescale=1./255,
                               rotation_range=15,
                               zoom_range=0.15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               horizontal_flip=True).flow_from_directory(trainpath,
                                                                         target_size=(imgsize, imgsize),
                                                                            batch_size=batchsize,
                                                                            class_mode='binary')

testdata = ImageDataGenerator(rescale=1./255).flow_from_directory(testpath,
                                                                      target_size=(imgsize, imgsize),
                                                                      batch_size=batchsize,
                                                                      class_mode='binary',
                                                                      shuffle=False)

#model creation 

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(imgsize, imgsize, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#model training

history = model.fit(
    traindata,
    validation_data=testdata,
    epochs=epoch,
    verbose=1
)

#plot wala part

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='test accuracy')
plt.title('Model Accuracy')
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.title('Model Loss')
plt.legend(['Train', 'Test'])
plt.show()

y_pred_prob = model.predict(testdata, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = testdata.classes
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
for i in range (2):
    for j in range (2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.show()
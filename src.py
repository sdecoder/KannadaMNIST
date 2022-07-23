'''Importing Data Manipulattion Moduls'''
import numpy as np
import pandas as pd

'''Seaborn and Matplotlib Visualization'''
import matplotlib.pyplot as plt
import seaborn as sns

'''Importing preprocessing libraries'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

'''Display markdown formatted output like bold, italic bold etc.'''

'''Importing tensorflow libraries'''
import tensorflow as tf

print(tf.__version__)
from keras import layers, models

'''Read in train and test data from csv files'''
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
sample_sub = pd.read_csv("input/sample_submission.csv")

'''Seting X and Y'''
y_train = train['label']
# Drop 'label' column
X_train = train.drop('label', axis=1)
X_test = test.drop('id', axis=1)

print('Input matrix dimension:', X_train.shape)
print('Output vector dimension:', y_train.shape)
print('Test data dimension:', X_test.shape)
images = train.iloc[:, 1:].values
images = images.astype(np.float)
# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

image_size = images.shape[1]
print('image_size => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))

'''Normalizing the data'''
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='uint8')
print(f"[trace] # 4.3 Reshape")
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

print(f"[trace] Set the random seed")
seed = 44
print(f"[trace] Split the train and the validation set for the fitting")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)

print(
	f"[trace] CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out")
model = models.Sequential()
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Dropout(0.25),
	tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Dropout(0.25),
	tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Dropout(0.25),
	tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Dropout(0.25),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(256, activation='relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()
'''
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	featurewise_center=False,  # set input mean to 0 over the dataset
	samplewise_center=False,  # set each sample mean to 0
	featurewise_std_normalization=False,  # divide input by std of the dataset
	samplewise_std_normalization=False,  # divide each input by its std
	zca_whitening=False,  # apply ZCA whitening
	rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
	zoom_range=0.1,  # Randomly zoom image
	width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=False,  # randomly flip images
	vertical_flip=False)  # randomly flip images

'''

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=1000, epochs=10, validation_data=(X_val, y_val), verbose=2)
results = model.predict(X_test)

'''select the indix with the maximum probability'''
results = np.argmax(results, axis=1)
sample_sub['label'] = results
sample_sub.to_csv('submission.csv', index=False)

'''
2022-02-26 22:34:38.168378: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8302
54/54 - 12s - loss: 0.4218 - accuracy: 0.8532 - val_loss: 0.0545 - val_accuracy: 0.9818 - 12s/epoch - 226ms/step
Epoch 2/10
54/54 - 9s - loss: 0.0483 - accuracy: 0.9860 - val_loss: 0.0285 - val_accuracy: 0.9922 - 9s/epoch - 162ms/step
Epoch 3/10
54/54 - 9s - loss: 0.0337 - accuracy: 0.9903 - val_loss: 0.0290 - val_accuracy: 0.9918 - 9s/epoch - 162ms/step
Epoch 4/10
54/54 - 9s - loss: 0.0255 - accuracy: 0.9926 - val_loss: 0.0209 - val_accuracy: 0.9942 - 9s/epoch - 162ms/step
Epoch 5/10
54/54 - 9s - loss: 0.0197 - accuracy: 0.9943 - val_loss: 0.0166 - val_accuracy: 0.9953 - 9s/epoch - 162ms/step
Epoch 6/10
54/54 - 9s - loss: 0.0151 - accuracy: 0.9954 - val_loss: 0.0174 - val_accuracy: 0.9945 - 9s/epoch - 162ms/step
Epoch 7/10
54/54 - 9s - loss: 0.0140 - accuracy: 0.9956 - val_loss: 0.0199 - val_accuracy: 0.9950 - 9s/epoch - 163ms/step
Epoch 8/10
54/54 - 9s - loss: 0.0116 - accuracy: 0.9966 - val_loss: 0.0193 - val_accuracy: 0.9953 - 9s/epoch - 163ms/step
Epoch 9/10
54/54 - 9s - loss: 0.0094 - accuracy: 0.9969 - val_loss: 0.0212 - val_accuracy: 0.9943 - 9s/epoch - 163ms/step
Epoch 10/10
54/54 - 9s - loss: 0.0097 - accuracy: 0.9968 - val_loss: 0.0170 - val_accuracy: 0.9962 - 9s/epoch - 171ms/step
'''

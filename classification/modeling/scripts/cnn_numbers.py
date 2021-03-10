# Simple CNN model for a dataset

# Import section
import numpy as np
import os
import errno
from scipy import misc
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import model_from_json

# Import (YOUR) dataset module
import dataset as service


# Create directories
def make_sure_path_exists(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

make_sure_path_exists('./models')


# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# Load data
# Load (YOUR) dataset with its respectively outputs
(X_train, y_train) = service.load_dataset('./npz/service_numbers_train.npz') 
(X_test, y_test) = service.load_dataset('./npz/service_numbers_test.npz') 

print('X_train.shape', X_train.shape)
plt.subplot(111)
plt.imshow(X_train[0])
plt.show()


# Normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0



# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print('num_classes: ', num_classes)



# Reshape to be [samples][channels][width][height]
# Set (YOUR) configuration 
X_train = X_train.reshape(X_train.shape[0], 1, 20, 30).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 20, 30).astype('float32')




# Create the model
model = Sequential()
model.add(Conv2D(32, (9, 9), input_shape=(1, 20, 30), activation='relu', padding='same'))
model.add(Dropout(0.4))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (6, 6), activation='relu', padding='same'))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))



# Serialize model to JSON 
model_json = model.to_json() 
with open('./models/model_numbers.json', 'w') as json_file:
  json_file.write(model_json)



# Compile model
epochs = 200
lrate = 0.01
decay = lrate/epochs
# Stochastic Gradient Descent
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())



# Checkpoint 
filepath = './models/model_numbers.h5'
#filepath = './models/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') 
callbacks_list = [checkpoint]



# Fit the model
# Be aware with the batch_size
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=50, callbacks=callbacks_list, verbose=0)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))






# Load json and create model 
json_file = open('./models/model_numbers.json', 'r')
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json) 

# Load weights into new model 
loaded_model.load_weights('./models/model_numbers.h5') 
print('Loaded model from disk')

# Evaluate loaded model on test data 
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
scores = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))







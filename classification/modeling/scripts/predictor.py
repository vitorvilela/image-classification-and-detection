# service predictor

from PIL import Image
import numpy as np
import os
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.optimizers import SGD
# Stochastic Gradient Descent
epochs = 200
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

# Import (YOUR) dataset module
import dataset as service

#################################### NUMBER ########################################

# Load (YOUR) dataset with its respectively outputs
(X_test, y_test) = service.load_dataset('./npz/service_numbers.npz')  

# Normalize inputs from 0-255 to 0.0-1.0
X_test = X_test.astype('float32')
X_test = X_test / 255.0

# One hot encode outputs
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print('\nnum_classes: ', num_classes)


# Reshape to be [samples][channels][width][height]
# Set (YOUR) configuration 
X_test = X_test.reshape(X_test.shape[0], 1, 20, 30).astype('float32')


# Load json and create model 
json_file = open('./models/model_numbers.json', 'r')
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json) 

# Load weights into new model 
loaded_model.load_weights('./models/model_numbers.h5') 
print('Loaded model_numbers from disk')

# Evaluate loaded model on test data 
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
scores = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))



#################################### LETTERS ########################################

(X_test, y_test) = service.load_dataset('./data/service_letters.npz') 

# Normalize inputs from 0-255 to 0.0-1.0
X_test = X_test.astype('float32')
X_test = X_test / 255.0

# Multiclass classification before
# encoding class values as integers
encoder = LabelEncoder()
encoder.fit(y_test)
encoded_y_test = encoder.transform(y_test)
encoder.fit(y_test)
encoded_y_test = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
y_test = np_utils.to_categorical(encoded_y_test)
num_classes = y_test.shape[1]
print('num_classes: ', num_classes)


# Reshape to be [samples][channels][width][height]
# Set (YOUR) configuration 
X_test = X_test.reshape(X_test.shape[0], 1, 20, 30).astype('float32')



# Load json and create model 
json_file = open('./models/model_letters.json', 'r')
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json) 

# Load weights into new model 
loaded_model.load_weights('./models/model_letters.h5') 
print('Loaded model_letters from disk')

# Evaluate loaded model on test data 
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
scores = loaded_model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))

print('\n')
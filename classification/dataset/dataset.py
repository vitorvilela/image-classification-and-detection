# Load libraries
import numpy as np
import os
from scipy import misc
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# Build dataset of images and targets into a .npz compressed file 
def build_dataset(img_path, img_extension, npz_path_name):

  list_of_images = []
  list_of_targets = []
    
  # Append images and targets into a list  
  for img_name in os.listdir(img_path):
    if img_name.endswith('.'+img_extension):     
      img_path_name = os.path.join(img_path, img_name)
      image = misc.imread(img_path_name)
      if image.ndim != 2:
        continue
      list_of_images.append(image)      
      list_of_targets.append(img_name.split('_')[0])
     
  # Save a .npz compressed file  
  np.savez_compressed(npz_path_name, X=list_of_images, y=list_of_targets)



# Load dataset from .npz compressed file
def load_dataset(npz_path_name):
          
  dataset = np.load(npz_path_name)
  # X is the images and y is the targets
  X, y = dataset['X'], dataset['y']
  dataset.close()
      
  # Returns a tuple of numpy arrays: (X, y)
  return (X, y)
    
  
  
# Plot dataset samples
def plot_dataset(npz_path_name):
  
  (X, y) = load_dataset(npz_path_name)

  # Create a grid of 3x3 images
  for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X[i])
  plt.show()
import os
import shutil
import errno


# Characters dictionary
char_dict = {}

# Split proportion
train_proportion = 0.8

# Open quantidade.txt
# Exclude first line
# Read lines in file
# Move to different folders acconding to split proportion


# Building characters dictionary
with open("./quantity.txt") as f:
  for line in f:
    if not line.startswith('Atualizado'):
      char_dict[line.split(':')[0]] = int(line.split(':')[1])


# Create directories
def make_sure_path_exists(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

make_sure_path_exists('./images/numbers/train/')
make_sure_path_exists('./images/numbers/test/')
make_sure_path_exists('./images/letters/train/')
make_sure_path_exists('./images/letters/test/')
    
    
# Split characters into train and test folders  
for img_name in os.listdir('./images'):
  
  print(img_name)  
  if img_name.endswith('.jpg'): 
    
    char_name = img_name.split('_')[0]
    char_value = img_name.split('_')[1].split('.')[0]
    
    if char_value == '':
      continue
    
    # Process numbers    
    if char_name <= '9':
      if int(char_value) < train_proportion*char_dict[char_name]:
        shutil.move('./images/'+img_name, './images/numbers/train/'+img_name)
      else:
        shutil.move('./images/'+img_name, './images/numbers/test/'+img_name)
          
    # Process letters    
    else:
      if int(char_value) <= train_proportion*char_dict[char_name]:
        shutil.move('./images/'+img_name, './images/letters/train/'+img_name)
      else:
        shutil.move('./images/'+img_name, './images/letters/test/'+img_name)   


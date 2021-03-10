import os
from PIL import Image

size = 704, 480

img_last_number = 0

for img_name in os.listdir("./raw"):
    
  file, ext = os.path.splitext(img_name)
  im = Image.open("./raw/"+img_name)
  
  img_last_number += 1
  print(img_number, im.format, im.size, im.mode, '\n')
  
  if im.size != size:
    out = im.resize(size)
  else: 
    out = im
  out.save('./images/'+str(img_last_number)+'.jpg', 'JPEG')

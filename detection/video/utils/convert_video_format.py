import os

# Root directory of the project
ROOT_DIR = os.getcwd()

# Videos directory
VIDEO_DIR = os.path.join(ROOT_DIR, 'videos')

# Directory to save images
OUT_DIR = os.path.join(ROOT_DIR, 'images')

IN_NAME = 'test.mp4'
VIDEO_NAME = 'test.mp4'

# Convert video from .DAV into .AVI
# This method will drop frames to achieve the desired speed
FACTOR = 10
SPEEDUP = 'setpts='+str(1/FACTOR)+'*PTS'
COMMAND = 'ffmpeg -i ' + os.path.join(VIDEO_DIR, IN_NAME) + ' -filter:v ' + SPEEDUP + ' ' + os.path.join(VIDEO_DIR, VIDEO_NAME)
os.system(COMMAND)

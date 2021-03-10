import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import profile


CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'coco_inference_graph.pb'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'models', MODEL_NAME)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')

# mscoco: 90, mscoco_modified: 1, mobez: 2, mobez_coco: 91 
NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def main(args):   

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    
    
    #profile.run('pool = Pool(args.num_workers, worker, (input_q, output_q))')
    pool = Pool(args.num_workers, worker, (input_q, output_q))

    # Inside this function there is a thread providing frames
    video_capture = WebcamVideoStream(src=args.video_source,
                                     width=args.width,
                                     height=args.height).start()
    
    #PATH_TO_FILE = os.path.join(CWD_PATH, 'rtsp://192.168.0.109:554/user=admin&password=admin&channel=1&stream=0.sdp?')
    #video_capture = WebcamVideoStream(src=PATH_TO_FILE,
    #                                  width=args.width,
    #                                  height=args.height).start()
           
    fps = FPS().start()

    while True: #fps._numFrames < 120:
		
		# Here the frames are read and placed into a Queue, which feeds a Pool
        frame = video_capture.read()
        input_q.put(frame)

        t = time.time()

        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', output_rgb)
        fps.update()

        #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    #print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    #print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()



def snapshot_trigger(looking, boxes, classes, scores, category_index, use_normalized_coordinates=True, max_boxes_to_draw=5, min_score_thresh=.5):
 
    # Boxes are ordered into nparray following its score value   
    for i in range(max_boxes_to_draw):  
        if scores[i] > min_score_thresh and category_index[classes[i]]['name'] == looking:      
           return boxes[i].tolist()

    return None 



def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection
    t1 = time.time()
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
        
    # Verify if both classes are detected and returns false or true
    # 'plate' or 'car'
    detectedPlate = snapshot_trigger('plate', np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, max_boxes_to_draw=10, min_score_thresh=.8)

    t2 = time.time()
    print(t2-t1)

    # Get object coordinates
    if detectedPlate != None:              
      print('\nplate', ' (ymin, xmin, ymax, xmax): ', detectedPlate)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np



def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count = {'CPU': 1})
        #sess = tf.Session(graph=detection_graph)
        sess = tf.Session(graph=detection_graph, config=config)
        
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        # Call detection
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))
        # Doesn't call detection
        #output_q.put(frame_rgb)

    fps.stop()
    sess.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Parser restricts the argument video_source be an integer
    # To work with files and URL, pass the string directly
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()
    
    #profile.run('main(args)')
    main(args)



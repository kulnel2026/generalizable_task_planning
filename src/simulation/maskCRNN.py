import tensorflow as tf
import numpy as np
import cv2
import pybullet as p
import pybullet_data

def load_model():
    model_dir = tf.keras.utils.get_file(
        'mask_rcnn_coco',
        'https://storage.googleapis.com/tfjs-models/savedmodel/pose-detection/mobilenet_v1_075_128/model.json',
        cache_dir='.', cache_subdir='models'
    )
    model = tf.saved_model.load(model_dir)
    return model

def run_inference_for_single_image(model, image):
    image_np = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    output_dict = model(input_tensor)
    
    
    output_dict = {key:value.numpy() for key,value in output_dict.items()}
    
    num_detections = int(output_dict['num_detections'][0])
    boxes = output_dict['detection_boxes'][0]
    classes = output_dict['detection_classes'][0]
    scores = output_dict['detection_scores'][0]
    
    return boxes, classes, scores, num_detections

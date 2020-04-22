import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import os
import sys

from PIL import Image
from tensorflow.lite.python import interpreter as interpreter_wrapper

# constants
IMG_HEIGHT = 416
IMG_WIDTH = 416

GRID_HEIGHT = 13
GRID_WIDTH = 13
BLOCK_SIZE = 32
NUM_BOXES_PER_BLOCK = 5
NUM_CLASSES = 80

THRESHOLD = 0.5
OVERLAP_THRESHOLD = 0.3

ANCHORS = [
    [0.57273, 0.677385], 
    [1.87446, 2.06253], 
    [3.33843, 5.47434], 
    [7.88282, 3.52778], 
    [9.77052, 9.16828]
]

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 
    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# index in results
R_LEFT         = 0
R_TOP          = 1
R_RIGHT        = 2
R_BOTTOM       = 3
R_CONFIDENCE_C = 4
R_CONFIDENCE   = 5
R_CLASS        = 6

def expit(x):
    return 1 / (1 + np.exp(-1 * x))

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def sort_results(e):
    return e[R_CONFIDENCE_C]

def check_result(data):
    results = []
    for row in range(GRID_HEIGHT):
        for column in range(GRID_WIDTH):
            for box in range(NUM_BOXES_PER_BLOCK):
                # x, y, w, h, object confidence, classes score...
                # one box = 5 + 80
                item = data[row][column]
                offset = (NUM_CLASSES+5) * box
                
                confidence = expit(item[offset+4])
                
                classes = item[offset+5+0:offset+5+NUM_CLASSES]
                classes = softmax(classes);
                
                detected_class = np.argmax(classes)
                max_class = classes[detected_class]
                
                confidence_in_class = max_class * confidence;
                
                if confidence_in_class > THRESHOLD:
                    x_pos = (column + expit(item[offset+0])) * BLOCK_SIZE;
                    y_pos = (row + expit(item[offset+1])) * BLOCK_SIZE;
                    w = (np.exp(item[offset+2]) * ANCHORS[box][0]) * BLOCK_SIZE
                    h = (np.exp(item[offset+3]) * ANCHORS[box][1]) * BLOCK_SIZE
                    
                    left = max(0, x_pos - w / 2)
                    top = max(0, y_pos - h / 2)
                    right = min(IMG_WIDTH - 1, x_pos + w / 2)
                    bottom = min(IMG_HEIGHT - 1, y_pos + h / 2)
                    
                    msg = '(%03d, %03d) (%03d, %03d) %03.2f %03.2f %s' % (int(left), int(top), int(right), int(bottom), confidence_in_class, confidence, COCO_CLASSES[detected_class])
                    print(msg)
                    results.append([left, top, right, bottom, confidence_in_class, confidence, COCO_CLASSES[detected_class]])   
    
    #sys.exit()
    
    # NMS?
    results.sort(reverse=True, key=sort_results)
    
    predictions = []
    best_prediction = results.pop(0)
    predictions.append(best_prediction)
    
    i = 0
    while True:
        if i >= len(results):
            break
        
        prediction = results.pop(0)
        overlaps = False
        
        for j in range(len(predictions)):
            previousPrediction = predictions[j]
            
            intersectProportion = 0.0
            primary = previousPrediction
            secondary = prediction
            
            if primary[R_LEFT] < secondary[R_RIGHT] and primary[R_RIGHT] > secondary[R_LEFT] and \
               primary[R_TOP] < secondary[R_BOTTOM] and primary[R_BOTTOM] > secondary[R_TOP]:

                intersection = max(0, min(primary[R_RIGHT], secondary[R_RIGHT]) - max(primary[R_LEFT], secondary[R_LEFT])) * \
                               max(0, min(primary[R_BOTTOM], secondary[R_BOTTOM]) - max(primary[R_TOP], secondary[R_TOP]))

                main = np.abs(primary[R_RIGHT] - primary[R_LEFT]) * np.abs(primary[R_BOTTOM] - primary[R_TOP])
                intersectProportion= intersection / main
            
            overlaps = overlaps or (intersectProportion > OVERLAP_THRESHOLD)
        
        if overlaps == False:
            predictions.append(prediction);
        
        i = i + 1
        
    for i in range(len(predictions)):
        print(predictions[i])
    return predictions
    
def get_input_data(img_file):
    image = Image.open(img_file)
    new_image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    n = np.array(new_image, dtype='float32')
    n = n / 255.0
    
    input_data = np.array([n], dtype='float32')
    return new_image, input_data

def arrange_point(item):
    x = []
    y = []
    
    left = int(item[R_LEFT])
    top = int(item[R_TOP])
    right = int(item[R_RIGHT])
    bottom = int(item[R_BOTTOM])
    
    x.append(left)
    y.append(top)
    
    x.append(right)
    y.append(top)
    
    x.append(right)
    y.append(bottom)
    
    x.append(left)
    y.append(bottom)
    
    x.append(left)
    y.append(top)

    return x, y

def select_image(i):
    images = ['art.jpg']
    return os.path.join('images', images[i])

def main():
    print('Tensorflow version is', tf.__version__)
    
    interpreter = tf.lite.Interpreter(model_path='tiny_yolo.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # set input
    image_index = 0
    image_file = select_image(image_index)
    
    new_image, input_data = get_input_data(image_file)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # run inference
    interpreter.invoke()
    
    # output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # process output
    data = output_data[0]
    predictions = check_result(data)
    
    # draw detection rectange on resized image
    plt.imshow(new_image, cmap=plt.cm.binary)
    
    colors = ['b', 'r']
    for i in range(len(predictions)):
        x_points, y_points = arrange_point(predictions[i])
        plt.plot(x_points, y_points, linestyle='-',color=colors[i%2], linewidth=1)
        
        hint = '%s %0.2f' % (predictions[i][R_CLASS], predictions[i][R_CONFIDENCE_C])
        plt.text(x_points[0], y_points[0] - 4, hint)
    
    plt.show()
                    

if __name__ == '__main__':
    main()

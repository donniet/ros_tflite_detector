#!/usr/bin/env python3

import argparse
from functools import partial

import numpy as np
import tflite_runtime.interpreter as tflite

import rospy
from sensor_msgs.msg import Image
from tflite_detector.msg import Detect, Box

def process(interpreter, input_details, output_details, publisher, image):
    dat = np.frombuffer(image.data, dtype=np.uint8)
    dat = dat.reshape(input_details[0]['shape'])

    interpreter.set_tensor(input_details[0]['index'], dat)
    interpreter.invoke()

    det = Detect()
    det.num = int(interpreter.get_tensor(output_details[3]['index']))
    det.scores = interpreter.get_tensor(output_details[2]['index'])
    #det.classes implement this 
    det.boxes = []
    detection_boxes = interpreter.get_tensor(output_details[0]['index'])
    if det.num > 0:
        for i in range(len(detection_boxes)):
            box = Box()
            box.top = detection_boxes[i][0]
            box.left = detection_boxes[i][1]
            box.bottom = detection_boxes[i][2]
            box.right = detection_boxes[i][3]
            det.boxes.append(box)

    publisher.publish(det)
        
    print('processing image, found: {}'.format(det.num))

def listener(args):
    if args.edgetpu:
        interpreter = tflite.Interpreter(
            model_path = args.model_path,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
    else:
        interpreter = tflite.Interpreter(
            model_path = args.model_path
        )

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('input: {}'.format(input_details))
    print('output: {}'.format(output_details))

    rospy.init_node('detector', anonymous=True)

    publisher = rospy.Publisher('detections', Detect, queue_size=100)

    rospy.Subscriber('rawFrames', Image, partial(process, interpreter, input_details, output_details, publisher))

    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='directory to saved_model', default='.')
    parser.add_argument('--edgetpu', help='load edgetpu library', type=bool, default=True)
    
    listener(parser.parse_args())

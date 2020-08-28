#!/usr/bin/env python3

import argparse
from functools import partial

import io
import numpy as np
import tflite_runtime.interpreter as tflite
import PIL.Image
import PIL.ImageDraw

import rospy
from sensor_msgs.msg import Image, CompressedImage
from tflite_detector.msg import Detect, Box

frame = 0

def process(interpreter, input_details, output_details, publisher, annotations, image):
    global frame
    dat = np.frombuffer(image.data, dtype=np.uint8)
    dat = dat.reshape(input_details[0]['shape'])

    interpreter.set_tensor(input_details[0]['index'], dat)
    interpreter.invoke()

    im = PIL.Image.fromarray(dat[0])
    draw = PIL.ImageDraw.Draw(im)
    

    det = Detect()
    det.num = int(interpreter.get_tensor(output_details[3]['index'])[0])
    det.scores = interpreter.get_tensor(output_details[2]['index'])[0][0:det.num]
    #det.classes implement this 
    det.boxes = []
    detection_boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    if det.num > 0:
        for i in range(det.num):
            box = Box()
            box.coords = detection_boxes[i]
            det.boxes.append(box)

            if det.scores[i] > 0.5:
                bb = np.array(detection_boxes[i])*320

                draw.rectangle((bb[1], bb[0], bb[3], bb[2]))

    with io.BytesIO() as output:
        im.save(output, format='JPEG')
        compressed = CompressedImage()
        compressed.header.stamp = rospy.Time.now()
        compressed.format = 'jpeg'
        compressed.data = output.getvalue()

#        with open('/tmp/frame{0:05d}.jpg'.format(frame), 'wb') as f:
#            f.write(output.getvalue())
#            frame = (frame + 1) % 10

    publisher.publish(det)
    annotations.publish(compressed)


        
    #print('processing image, found: {}'.format(det.num))

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
    annotations = rospy.Publisher('annotations', CompressedImage, queue_size=100)

    rospy.Subscriber('rawFrames', Image, partial(process, interpreter, input_details, output_details, publisher, annotations))

    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='directory to saved_model', default='.')
    parser.add_argument('--label_map', help='path to label_map file', default='label_map.pbtxt')
    parser.add_argument('--edgetpu', help='load edgetpu library', type=bool, default=True)
    
    listener(parser.parse_args())

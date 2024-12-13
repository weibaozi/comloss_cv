#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from model import myModel


def main():
    model_path = 'model.pth'
    mymodel=myModel(model_path, 'cpu')
    #open image file
    image = cv2.imread('1.jpg')
    output = mymodel.predict(image)
    print(output.sigmoid())

if __name__ == '__main__':
    main()

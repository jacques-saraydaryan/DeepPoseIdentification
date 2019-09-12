#! /usr/bin/env python
__author__ ='Jacques Saraydaryan, Nina Tubau Ribera, Pierre Assemat'

import rospy
import cv2
import os
import numpy as np
import json
import argparse

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from openpose_ros_msgs.msg import Persons, PersonDetection
from openpose_ros_srvs.srv import DetectPeoplePoseFromImg

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


from prediction_bis import Prediction

class ProcessFocusUnfocusData():

    def __init__(self):
        ''' Initialise ProcessFocusUnfocusData class by subscribing to 2 Ros channels:
        Persons: joint Pose stream  
        Image: Raw image stream
        '''
        self._bridge = CvBridge()
        self.currentImage = None

        rospy.init_node('ProcessFocusUnfocusData', anonymous=True)

        rospy.Subscriber("/openpose/pose", Persons, self.processOpenPoseJsonData)
        rospy.Subscriber("/openpose/image_raw", Image, self.processOpenPoseImgData)

    def processOpenPoseJsonData(self, personObject):
        '''Callback run each time a new json position joints is received
        '''
        # Callback when openpose output received
        rospy.loginfo('[OPENPOSE_JSON] received data:')

        # Preprocess stream
        h = self.currentImage.shape[0]
        w = self.currentImage.shape[1]
        
        json_positions = []

        # Reconstruct a python dict to add image_size on each person
        for i in range(len(personObject.persons)):
            bodyParts = []
            for j in range(len(personObject.persons[i].body_part)):
                bodyParts.append({
                    'x': personObject.persons[i].body_part[j].x,
                    'y': personObject.persons[i].body_part[j].y,
                    'confidence': personObject.persons[i].body_part[j].confidence
                })
            json_positions.append({
                "body_part": bodyParts,
                "face_landmark": personObject.persons[i].face_landmark,
                "image_size": {"width": w, "height": h}
            })

        # Predict classes
        predictionObject = Prediction()

        data = predictionObject.preprocess(json_positions)

        predictions = predictionObject.predict(data)

        predictions = predictionObject.predictClasses(data)

        for i in range(len(predictions)):
            print('Person %s' %str(i+1) + ' - ' + predictionObject.LABEL[predictions[i][0]])
            self.postProcess(json_positions[i], int(predictions[i]))
     


    def postProcess(self, json_positions, prediction):
        
        #green for focus and red for distract
        color = [(0,255,0),(0,0,255)]

        x_ear = json_positions['body_part'][16]['x']
        y_ear = json_positions['body_part'][16]['y']

        x_nose = json_positions['body_part'][0]['x']
        y_nose = json_positions['body_part'][0]['y']

        x_neck = json_positions['body_part'][1]['x']
        y_neck = json_positions['body_part'][1]['y']

        x_0 = x_ear
        y_0 = y_ear + (y_nose-y_neck)

        width = 2*(x_nose-x_ear)
        height = 2*(y_0-y_ear)

        cv2.rectangle(self.currentImage, (x_0,y_0), (x_0+width, y_0-height), color[prediction], 2)
        cv2.imshow("Labelised image", self.currentImage)
        cv2.waitKey(1)



    def processOpenPoseImgData(self, data):
        '''Callback run each time a raw image is received
        '''
        frame = self._bridge.imgmsg_to_cv2(data, 'bgr8')
        rospy.loginfo('[OPENPOSE_IMAGE] received img:')
        self.currentImage = frame


if __name__ == '__main__':

    try:
        ProcessFocusUnfocusData()
    except rospy.ROSInterruptException:
        pass

    # Spin
    rospy.spin()
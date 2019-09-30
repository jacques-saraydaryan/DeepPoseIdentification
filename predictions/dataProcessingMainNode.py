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
from openpose_ros_msgs.msg import Persons, PersonDetection,PersonsImgPose
from openpose_ros_srvs.srv import DetectPeoplePoseFromImg

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


from prediction import Prediction

class ProcessFocusUnfocusData():

    def __init__(self):
        ''' Initialise ProcessFocusUnfocusData class by subscribing to 2 Ros channels:
        Persons: joint Pose stream  
        Image: Raw image stream
        '''
        self._bridge = CvBridge()

        rospy.init_node('ProcessFocusUnfocusData', anonymous=True)

        rospy.Subscriber("/openpose/pose_and_img", PersonsImgPose, self.processOpenPoseJsonData)

        self._output_image_pub = rospy.Publisher("/deep_pose_identification/image_output", Image)

    def processOpenPoseJsonData(self, data):
        '''Callback run each time a new json position joints is received
        '''
        # Callback when openpose output received
        rospy.loginfo('[OPENPOSE_JSON] received data:')

        ### TODO: Fix the publisher of '/openpose/pose_and_img' supposed to contain both the json_position object and the input image

        #currentImage = data.image
        currentImage= self._bridge.imgmsg_to_cv2( data.image, desired_encoding = "bgr8")
        personObject = data.persons

        # Preprocess stream
        h = currentImage.shape[0]
        #h = currentImage.height
        #w = currentImage.width
        w = currentImage.shape[1]

        json_positions = []

        # Reconstruct a python dict to add imageSize on each person
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
        self.predictionObject = Prediction()
        data = self.predictionObject.preprocess(json_positions)

        # If no person is detected, just return the image raw
        if len(data):
            predictions = self.predictionObject.predict(data)

            predictions = self.predictionObject.predictClasses(data)

            for i in range(len(predictions)):
                print('Person %s' %str(i+1) + ' - ' + self.predictionObject.LABEL[predictions[i][0]])
                self.postProcess(json_positions[i], int(predictions[i]), currentImage)

        ros_msg_image = self._bridge.cv2_to_imgmsg(currentImage, 'bgr8')
        self._output_image_pub.publish(ros_msg_image)


    def postProcess(self, json_positions, prediction, currentImage):

        #green for focus and red for distract
        color = [(0,255,0),(0,0,255)]
        x=[]
        y=[]

        for i in range(len(json_positions['body_part'])):
            x_value = json_positions['body_part'][i]['x']
            y_value = json_positions['body_part'][i]['y']
            if(x_value!=0):
                x.append(x_value)
            if(y_value!=0):
                y.append(y_value)

        x_min = min (x)
        x_max = max(x)
        y_min = min(y)
        y_max =  max(y)   

        width = x_max - x_min 
        height = y_max - y_min

        cv2.rectangle(currentImage, (x_min, y_min - height/10), (x_max, y_max), color[prediction], 2)


if __name__ == '__main__':

    try:
        ProcessFocusUnfocusData()
    except rospy.ROSInterruptException:
        pass

    # Spin
    rospy.spin()

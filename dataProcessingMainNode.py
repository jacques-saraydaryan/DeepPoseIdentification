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

from prediction_bis import Prediction

class ProcessFocusUnfocusData():

    def __init__(self):
        self._bridge = CvBridge()
        self.currentImage = None

        rospy.init_node('ProcessFocusUnfocusData', anonymous=True)

        rospy.Subscriber("/openpose/pose", Persons, self.processOpenPoseJsonData)
        rospy.Subscriber("/openpose/image_raw", Image, self.processOpenPoseImgData)

    def processOpenPoseJsonData(self, personObject):
        # Callback when openpose output received
        rospy.loginfo('[OPENPOSE_JSON] received data:')

        # Preprocess stream
        h = self.currentImage.shape[0]
        w = self.currentImage.shape[1]
        
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
                "imageSize": {"width": w, "height": h}
            })

        # Predict classes
        predictionObject = Prediction()
        data = predictionObject.preprocess(json_positions)
        predictions = predictionObject.predict(data)
        print(predictions)

        predictions = predictionObject.predictClasses(data)
        for i in range(len(predictions)):
            print('Person %s' %str(i+1) + ' - ' + predictionObject.LABEL[predictions[i][0]])

    def processOpenPoseImgData(self, data):
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
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
from openpose_ros_srvs.srv import DetectPeoplePoseFromImg

# Get argument parser
parser = argparse.ArgumentParser(description='Read images and convert into json file with OpenPose positions')
parser.add_argument('pathInput', type=str, default = './../imageDataset', help='Enter input folder with images')
parser.add_argument('pathOutput', type=str, default = './../openPoseDataset', help='Enter output path where to save json files')

# Convert ROS position object into python dict
def ConvertRes(res, w, h):
    results_list = []
    for i in range(len(res)):
        body_part = [{
            'part_id': res[i].body_part[j].part_id,
            'x': res[i].body_part[j].x,
            'y': res[i].body_part[j].y,
            'confidence': res[i].body_part[j].confidence
        } for j in range(len(res[i].body_part))]

        results_list.append({
            'body_part': body_part,
            'face_landmark': res[i].face_landmark,
            'imageSize': { 'width': w, 'height': h }
        })

    return results_list

# Load image, detect human position and save results as json file
def LoadImg(pathOutput, pathInput, _bridge):
    for image in os.listdir(pathInput):
        image_path = os.path.join(pathInput, image)

        if (os.path.isfile(image_path)):
            img_loaded = cv2.imread(image_path)
            w, h = img_loaded.shape[:2]
            msg_im = _bridge.cv2_to_imgmsg(img_loaded, encoding="bgr8")

            # Call service to learn people pose
            rospy.wait_for_service('people_pose_from_img')

            try:
                detect_from_img_srv = rospy.ServiceProxy('people_pose_from_img', DetectPeoplePoseFromImg)
                resp = detect_from_img_srv(msg_im)

                # Save position data as json file
                json_name = os.path.splitext(pathOutput + '/' + str(image))[0] + '.json'
                print(json_name)
                results = ConvertRes(resp.personList.persons, w, h)

                with open(json_name, 'w') as f:
                    f.write(json.dumps(results))
                    f.close()

            except (rospy.ServiceException, e):
                print("Service call failed: %s"%e)


def LoadImgAndPublish(pathInput, pathOutput):
    inputFolderName = pathInput.split('/')[-2]
    outputFolderName = pathOutput.split('/')[-2]

    _bridge = CvBridge()
    rospy.loginfo('media_folder:' + str(pathInput))

    rospy.init_node('LoadAndPublishImg', anonymous=True)

    # Get hierarchy of input path to copy on output path
    print('Loading images...')
    for folder in [x[0] for x in os.walk(pathInput)]:
        if (folder.endswith('distract') or folder.endswith('focus')):
            splitted = folder.split("/")[-2:]
            separator = '/'
            label_folder = pathOutput + separator.join(splitted)

            # Create folder if it does not exist
            if not(os.path.exists(label_folder)):
                source_folder = separator.join(label_folder.split("/")[:-1])
                if not(os.path.exists(source_folder)):
                    os.mkdir(source_folder)
                os.mkdir(label_folder)
            fullPathInput = label_folder.replace(outputFolderName, inputFolderName)
            LoadImg(label_folder, fullPathInput, _bridge)

    

if __name__ == '__main__':
    args = parser.parse_args()
    pathInput = args.pathInput
    pathOutput = args.pathOutput

    try:
        LoadImgAndPublish(pathInput, pathOutput)
    except rospy.ROSInterruptException:
        pass

    # Spin
    rospy.spin()
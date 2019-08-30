#import rospy

import processsing
import RosOpenPoseFiles


## Start detection chain for training

pathInput = './imageDataset'
pathOutput = './openPoseDataset'

# Detect human position using OpenPose and save result in json file tree structure
try:
    RosOpenPoseFiles.LoadImgAndPublish(pathInput, pathOutput)
except rospy.ROSInterruptException:
    pass

# Concat all the positions data into a single array and save it as pickle file
processsing.createInputMatrix(pathOutput)

# Construct the Neural Network classifier and start the learning phase
# TODO

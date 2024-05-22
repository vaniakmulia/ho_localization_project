#!/usr/bin/env python3

'''
    Node to detect ArUco marker from the RealSense camera mounted on the turtlebot.

    This node subscribes to the topics that contain the camera image as well as camera intrinsics.
    From the camera image, this node detects ArUco markers using OpenCV.
    When a marker is detected, this node estimate the pose of the marker w.r.t. camera.
    The pose is then converted into range to be used to update the EKF.
'''

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

from ho_localization_project.msg import ArucoRange # a newly created ROS msg

class ArUcoDetector:
    def __init__(self) -> None:
        self.image = np.zeros((1,1))
        self.camera_intrinsics = np.zeros((3,3))
        self.camera_distortions = np.zeros((5,))

        # ArUco marker data
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_marker_length = 0.150 #m

        # Initialize ArUco range list
        self.aruco_range_list = []

        # Initialize marker ID list
        self.marker_ids = None

        # Subscribers
        # for simulation
        # self.image_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/image_color", Image, self.image_callback)
        # self.intrinsics_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/camera_info", CameraInfo, self.intrinsics_callback)

        # for real robot
        self.image_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/image_raw/compressed", CompressedImage, self.image_callback)
        self.intrinsics_sub = rospy.Subscriber("/turtlebot/kobuki/realsense/color/camera_info", CameraInfo, self.intrinsics_callback)

        # Publishers
        self.range_pub = rospy.Publisher("/aruco_range",ArucoRange, queue_size=1)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(2), self.send_messages)

    def image_callback(self,image_msg):
        # Convert Image message into image
        # self.encoding = image_msg.encoding
        bridge = CvBridge()
        self.image = bridge.compressed_imgmsg_to_cv2(image_msg)

        marker_corners, self.marker_ids = self.detect_aruco()

        if self.marker_ids is not None:
            self.aruco_range_list = self.estimate_aruco_range(marker_corners, self.marker_ids)

    def detect_aruco(self):
        # Marker Detection
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, aruco_params)
        marker_corners, marker_ids, _ = detector.detectMarkers(self.image)

        return marker_corners, marker_ids
    
    def estimate_aruco_range(self, marker_corners, marker_ids):
        # Initialize a list of 3D corners of the marker (for pose estimation)
        object_points = np.zeros((4,1,3))
        object_points[0] = np.array((-self.aruco_marker_length/2, self.aruco_marker_length/2, 0)).reshape((1,3))
        object_points[1] = np.array((self.aruco_marker_length/2, self.aruco_marker_length/2, 0)).reshape((1,3))
        object_points[2] = np.array((self.aruco_marker_length/2, -self.aruco_marker_length/2, 0)).reshape((1,3))
        object_points[3] = np.array((-self.aruco_marker_length/2, -self.aruco_marker_length/2, 0)).reshape((1,3))

        # Initialize a list of marker range
        marker_range_list = []

        # Loop for all detected markers
        for i in range(len(marker_ids)):
            # Estimate the pose of the marker
            _, _, tvec = cv2.solvePnP(object_points, marker_corners[i], self.camera_intrinsics, self.camera_distortions)

            # print("id =", marker_ids[i])
            # print("tvec = ",tvec)

            # Convert into range
            marker_range = np.linalg.norm(tvec)
            marker_range_list.append(marker_range)

        return marker_range_list

        
    def intrinsics_callback(self,intrinsics_msg):
        # save the camera intrinsics only once (because it doesn't change over time)
        if not self.camera_intrinsics.any():
            self.camera_intrinsics = np.array(intrinsics_msg.K).reshape((3,3))
            self.camera_distortions = np.array(intrinsics_msg.D)

    def send_messages(self,event):
        if self.marker_ids is not None:
            try:
                # Send the list of ArUco range
                range_msg = ArucoRange()
                range_msg.id = list(self.marker_ids.flatten())
                range_msg.range = self.aruco_range_list

                self.range_pub.publish(range_msg)
            except rospy.ROSSerializationException as e:
                rospy.logwarn(e)


if __name__=='__main__':
    rospy.init_node('aruco_pose_to_range', anonymous=False) # initialize the node
    node = ArUcoDetector()

    rospy.spin()

# Hands-On Localization Project: FEKFSLAM with ArUco Range Observation

This package is made for Hands-On Localization project.

## Overview of Project

This project focuses on building a SLAM system for the Kobuki Turtlebot using EKF, integrating data from wheel encoders, IMU, and ArUco marker-based range observations to estimate the position and orientation of the Turtlebot. Wheel encoder readings are used to predict the pose of the turtlebot by dead-reckoning. Meanwhile, the orientation readings from IMU provide updates for the estimated pose of the turtlebot, improving the estimation. Furthermore, the algorithm incorporates feature-based updates from ArUco range observations (i.e., distance towards the observed ArUco markers), provided by an RGB camera (RealSense).

## Contents

This package contains:

* `localization_node.py`: node to localize a turtlebot with EKF, using wheel encoders for Prediction and yaw reading from IMU magnetometer for Update.

* `controller_node.py`: node to convert Twist into wheel velocities to control the turtlebot.

* `aruco_detector_node.py`: node to detect ArUco markers from RealSense camera and publish the range and the ID of detected markers.

Furthermore, this package also contains a custom ROS message, `ArucoRange`, which is used by the nodes in this package.

## Running the Package

To run the whole program with the turtlebot, first connect to the turtlebot and run `kobuki_mobile_base.launch` and `kobuki_sensors.launch` in the `turtlebot` package through ssh. Then, run the following command in the terminal:

```bash
roslaunch ho_localization_project launch_project.launch
```

This launch file will run all nodes in this package.

## Extra: Teleop

A nice teleop interface can be launched by typing this command line in the terminal:

```bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 
```

To run this, you need to have `teleop_twist_keyboard` installed. To install it, simply run

```bash
sudo apt-get install ros-noetic-teleop-twist-keyboard
```

For more information, refer to this [link](https://wiki.ros.org/teleop_twist_keyboard).

## Additional Documentation

The full report for this project can be found in [this link](https://drive.google.com/file/d/1Uq5TW2t79Qe-zPK_4U3bIB_kH5mAA2v_/view?usp=sharing).

A demo video of this project can be found in [this link](https://drive.google.com/file/d/1PYfLI1RPYdnKquDs4l_yUMtAEI38EcoM/view?usp=sharing).

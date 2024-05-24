# Hands-On Localization Project: FEKFSLAM with ArUco Range Observation

This package is made for Hands-On Localization project.

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
# Hands-On Localization Project

This package is made for Hands-On Localization project.

## Contents

This package contains:

* `localization_node.py`: node to localize a turtlebot with EKF, using wheel encoders for Prediction and yaw reading from IMU magnetometer for Update.

* `controller_node.py`: node to convert Twist into wheel velocities to control the turtlebot.

* `aruco_detector_node.py`: node to detect ArUco markers from RealSense camera and publish the range and the ID of detected markers.

## Dependencies

This package contains a custom message. This message can be found in this [repo](https://github.com/vaniakmulia/ho_custom_msgs) (Note: currently this is a private repo!). This package must be added to your `catkin_ws` (or the same workspace as this package).

Run the command lines below on a terminal

```bash
cd catkin_ws/src
git clone git@github.com:vaniakmulia/ho_custom_msgs.git
```

After adding the package, you have to build the package (using `catkin build`) and re-source the workspace.

## Running the Package

### Basic Functionalities: Localization and Controller

To run the package with Stonefish simulator, first run the simulator:

```bash
roslaunch turtlebot_simulation turtlebot_basic.launch 
```

> NOTE: You can also run other launch files in the `turtlebot_simulation` package to launch a different environment.

Then, there are several options:

To run only the localization node, run this command in another terminal:

```bash
rosrun ho_localization_project localization_node.py
```

And to test the localization, you can publish wheel velocities to the ROS topic `/turtlebot/kobuki/commands/wheel_velocities`. This topic accepts `Float64MultiArray` message, with a format = [vl,vr]. Another method to control the robot is to run the controller node:

```bash
rosrun ho_localization_project controller_node.py
```

This node allows for controlling the turtlebot using `Twist` message, and thus the robot can be controlled by teleop or with graphical tools such as _rqt_robot_steering_.

Alternatively, to run the localization node and controller node together, you can run the launch file included in this package. To do so, run the following command in another terminal:

```bash
roslaunch ho_localization_project localization_controller.launch
```

### ArUco Detector

To detect ArUco markers, run the `aruco_detector_node.py` with the command line below.

```bash
rosrun ho_localization_project aruco_detector_node.py
```

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

<sup>
Last updated by Vania - 18/04/2024
</sup>
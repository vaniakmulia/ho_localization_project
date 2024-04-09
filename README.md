# Turtlebot Localization Package

This package contains some nodes to localize a turtlebot with EKF, using wheel encoders for Prediction and yaw reading from IMU magnetometer for Update.

## Running the Package

To run the package with Stonefish simulator, first run the simulator:

```bash
roslaunch turtlebot_simulation turtlebot_basic.launch 
```

Then, there are several options:

To run only the localization node, run this command in another terminal:

```bash
rosrun turtlebot_localization localization_node.py
```

And to test the localization, you can publish wheel velocities to the ROS topic `/turtlebot/kobuki/commands/wheel_velocities`. This topic accepts `Float64MultiArray` message, with a format = [vl,vr]. Another method to control the robot is to run the controller node:

```bash
rosrun turtlebot_localization controller_node.py
```

This node allows for controlling the turtlebot using `Twist` message, and thus the robot can be controlled by teleop or with graphical tools such as `rqt_robot_steering` (NOTE: `rqt_robot_steering` is still not tested yet).

Alternatively, to run the localization node and controller node together, you can run the launch file included in this package. To do so, run the following command in another terminal:

```bash
roslaunch turtlebot_localization localization_controller.launch
```

<sup>
Last updated by Vania - 09/04/2024
</sup>
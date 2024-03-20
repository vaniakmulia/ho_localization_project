#!/usr/bin/env python3

import rospy
from turtlesim.msg import Pose
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
import tf
import math

class TurtlebotLocalization:
    def __init__(self) -> None:
        # initialize robot pose
        self.x = 0
        self.y = 0
        self.yaw = 0

        # initialize wheel velocities
        self.vl = 0 # left wheel
        self.vr = 0 # right wheel

        # initialize imu data
        self.orientation = 0 # from magnetometer

        # flag attribute to indicate start of localization
        self.start = False

        # add more attributes later

        # Subscribers
        self.states_sub = rospy.Subscriber("/turtlebot/joint_states",JointState,self.states_callback)
        self.imu_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/imu_data",Imu,self.imu_callback)

        # Publishers
        self.vel_pub = rospy.Publisher("/turtlebot/kobuki/commands/wheel_velocities",Float64MultiArray, queue_size=1)

        # subscriber callbacks
        def states_callback(self,states_msg):
            if self.start:
                # assign the wheel encoder values
                if states_msg.name == "turtlebot/kobuki/wheel_right_joint":
                    self.vr = states_msg.velocity

        def imu_callback(self,imu_msg):
            # assign the initial orientation to start
            if not self.start:
                _,_,self.yaw = tf.transformations.euler_from_quaternion([imu_msg.orientation.x, 
                                                              imu_msg.orientation.y,
                                                              imu_msg.orientation.z,
                                                              imu_msg.orientation.w])
                self.start = True # start the localization
            else:
                pass


if __name__=='__main__':
    rospy.init_node('move_to_point', anonymous=True) # initialize the node
    node = TurtlebotLocalization() # a newly-created class

    rospy.spin()
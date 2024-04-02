#!/usr/bin/env python3

import rospy
from turtlesim.msg import Pose
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
import tf
from math import sin,cos
import time

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

        # flag attribute to indicate left and right wheel encoder reading available
        self.right_wheel_flag = False
        self.left_wheel_flag = False

        # robot properties
        self.wheelbase = 0 # where do we get this??

        # add more attributes later

        # Subscribers
        self.states_sub = rospy.Subscriber("/turtlebot/joint_states",JointState,self.states_callback)
        self.imu_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/imu_data",Imu,self.imu_callback)

        # Publishers
        self.odom_pub = rospy.Publisher("/turtlebot/kobuki/base_footprint",Odometry, queue_size=1)
        self.tf_pub = rospy.Publisher("/tf",TFMessage, queue_size=1)
        self.vel_pub = rospy.Publisher("/turtlebot/kobuki/commands/wheel_velocities",Float64MultiArray, queue_size=1)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(0.1), self.send_odom)


    # subscriber callbacks
    def states_callback(self,states_msg):
        if self.start:
            # assign the wheel encoder values
            if states_msg.name == "turtlebot/kobuki/wheel_right_joint":
                self.vr = states_msg.velocity
                self.right_wheel_flag = True
                self.right_wheel_prev_time = self.right_wheel_time
                self.right_wheel_time = time.time()
            elif states_msg.name == "turtlebot/kobuki/wheel_left_joint":
                self.vl = states_msg.velocity
                self.left_wheel_flag = True
                self.left_wheel_prev_time = self.left_wheel_time
                self.left_wheel_time = time.time()
            
            if self.right_wheel_flag and self.left_wheel_flag:
                self.x, self.y, self.yaw = self.dead_reckoning(self.vr,self.vl)

    def imu_callback(self,imu_msg):
        # start the localization when the node receives the first IMU reading
        if not self.start:
            self.start = True # start the localization
        
        # update the yaw from the magnetometer
        _,_,self.yaw = tf.transformations.euler_from_quaternion([imu_msg.orientation.x, 
                                                            imu_msg.orientation.y,
                                                            imu_msg.orientation.z,
                                                            imu_msg.orientation.w])
    
    def dead_reckoning(self,vr,vl):
        # set time interval
        # previous time = the later timestamp between the 2 wheels
        ti_1 = max(self.left_wheel_prev_time,self.right_wheel_prev_time)

        # current time = now
        ti = time.time()

        # time step
        dt = ti - ti_1

        # displacement of each wheel within the time interval
        dl = vl*dt
        dr = vr*dt

        # linear and angular displacement of robot
        d = (dl+dr)/2
        dtheta = (dr-dl)/self.wheelbase

        # predict robot pose
        thetak = self.yaw + dtheta
        xk = self.x + d*cos(thetak)
        yk = self.y + d*sin(thetak)

        return xk,yk,thetak

    def send_odom(self,event):
        # publish Odometry message
        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0

        yaw_quaternion = tf.transformations.quaternion_from_euler(0,0,self.yaw)
        odom_msg.pose.pose.orientation.x = yaw_quaternion[0]
        odom_msg.pose.pose.orientation.y = yaw_quaternion[1]
        odom_msg.pose.pose.orientation.z = yaw_quaternion[2]
        odom_msg.pose.pose.orientation.w = yaw_quaternion[3]

        self.odom_pub.publish(odom_msg)



if __name__=='__main__':
    rospy.init_node('turtlebot_localization', anonymous=True) # initialize the node
    node = TurtlebotLocalization() # a newly-created class

    rospy.spin()
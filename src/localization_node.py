#!/usr/bin/env python3

'''
    Node to perform localization using EKF.
'''

import rospy
from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
import tf
import time

from filter import *
from ho_custom_msgs.msg import ArucoRange # a newly created ROS msg

class TurtlebotLocalization:
    def __init__(self) -> None:
        # initialize robot pose
        self.x = 0
        self.y = 0
        self.yaw = 0

        # initialize robot pose covariance
        self.Pk = np.zeros((3,3))

        # initialize wheel angular velocities
        self.wl = 0 # left wheel
        self.wr = 0 # right wheel

        # initialize encoder covariance
        self.left_wheel_cov = 0.0 # left wheel
        self.right_wheel_cov = 0.0 # right wheel
        self.Re = np.diag([self.left_wheel_cov,self.right_wheel_cov])

        # initialize imu data
        self.orientation = 0 # from magnetometer

        # flag attribute to indicate start of localization
        self.start = False

        # encoder timestamp
        self.left_wheel_time = time.time()
        self.right_wheel_time = time.time()

        # flag attribute to indicate left and right wheel encoder reading available
        self.right_wheel_flag = False
        self.left_wheel_flag = False

        # robot properties
        self.wheel_radius = 0.035 #m
        self.wheelbase = 0.235 #m

        # feature-related states
        self.feature_states = [] # coordinate of features already in the state vector
        self.feature_states_id = [] # ArUco ID of features already in state vector

        # feature observations
        self.feature_observation_ranges = {} # dict of ranges for trilateration
        self.feature_observation_points = {} # dict of observation points for trilateration

        # instantiate filter
        self.filter = EKF(np.array([self.x,self.y,self.yaw]).reshape(3,1),self.Pk)

        # Subscribers
        self.encoder_sub = rospy.Subscriber("/turtlebot/joint_states",JointState,self.encoder_callback)
        self.imu_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/imu_data",Imu,self.imu_callback)
        self.feature_sub = rospy.Subscriber("/aruco_range",ArucoRange,self.feature_callback)

        # Publishers
        self.odom_pub = rospy.Publisher("/odom",Odometry, queue_size=1)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(0.1), self.send_messages)


    ## Subscriber callbacks ###################
    def encoder_callback(self,states_msg):
        if self.start:
            # assign the wheel encoder values
            if states_msg.name[0] == "turtlebot/kobuki/wheel_right_joint":
                self.wr = states_msg.velocity[0]
                self.right_wheel_cov = np.deg2rad(3) #guess
                self.right_wheel_flag = True
                self.right_wheel_prev_time = self.right_wheel_time
                self.right_wheel_time = time.time()
            elif states_msg.name[0] == "turtlebot/kobuki/wheel_left_joint":
                self.wl = states_msg.velocity[0]
                self.left_wheel_cov = np.deg2rad(3) #guess
                self.left_wheel_flag = True
                self.left_wheel_prev_time = self.left_wheel_time
                self.left_wheel_time = time.time()
            
            if self.right_wheel_flag and self.left_wheel_flag:
                # Pre-processing for Prediction
                # rospy.loginfo("Performing prediction.")
                xk_1 = np.array([self.x,self.y,self.yaw]).reshape(3,1)
                Pk_1 = self.Pk
                self.Re = np.diag([self.left_wheel_cov,self.right_wheel_cov])

                # Convert encoder reading to odometry displacement
                uk,Qk = self.encoder_to_displacement(self.wl,self.wr,self.Re)

                # Perform Prediction
                xk_bar,Pk_bar = self.filter.Prediction(xk_1,Pk_1,uk,Qk)

                # Save Prediction result
                self.x = xk_bar[0,0]
                self.y = xk_bar[1,0]
                self.yaw = xk_bar[2,0]
                self.Pk = Pk_bar

                # Reset the flags
                self.right_wheel_flag = False
                self.left_wheel_flag = False

    def imu_callback(self,imu_msg):        
        # Get the yaw reading from the magnetometer
        _,_,yaw = tf.transformations.euler_from_quaternion([imu_msg.orientation.x, 
                                                            imu_msg.orientation.y,
                                                            imu_msg.orientation.z,
                                                            imu_msg.orientation.w])
        
        # Start the localization when the node receives the first IMU reading
        if not self.start:
            self.yaw = yaw # store the first IMU yaw reading as the initial yaw value
            self.start = True # start the localization
        else:
            # Pre-processing for Update
            # rospy.loginfo("Updating with IMU.")
            zk = np.array([[yaw]])
            Rk = np.array([[imu_msg.orientation_covariance[8]]])
            Hk = np.array([[0,0,1]])
            Vk = np.eye(1)

            xk_bar = np.array([self.x,self.y,self.yaw]).reshape(3,1)
            Pk_bar = self.Pk

            # Perform Update
            xk,Pk = self.filter.Update(xk_bar,Pk_bar,zk,Rk,Hk,Vk)

            # Save Update results
            self.x = xk[0,0]
            self.y = xk[1,0]
            self.yaw = xk[2,0]
            self.Pk = Pk

    def feature_callback(self,range_msg):
        marker_id = list(range_msg.id)
        marker_range = list(range_msg.range)

        for n in range(len(marker_id)):
            id = marker_id[n]
            # only add features to the states if it still doesn't exist in the states
            if id not in self.feature_states_id:
                # store the ranges and observation points for trilateration
                if id not in self.feature_observation_ranges.keys(): # marker ID observed for the first time
                    self.feature_observation_ranges[id] = [marker_range[n]]
                    self.feature_observation_points[id] = [(self.x,self.y)]
                else:
                    self.feature_observation_ranges[id].append(marker_range[n])
                    self.feature_observation_points[id].append((self.x,self.y))
        
        # Perform trilateration
        for id in self.feature_observation_ranges.keys():
            # check if the marker already has 3 observations
            if len(self.feature_observation_ranges[id]) == 3:
                xf,yf = self.trilateration(self.feature_observation_ranges[id],self.feature_observation_points[id])
                # if trilateration succeeds, add the feature to the states
                if xf and yf:
                    self.feature_states.append([xf,yf])
                    self.feature_states_id.append(id)
                    
                    # remove it from the list of observations to trilaterate
                    del self.feature_observation_ranges[id]
                    del self.feature_observation_points[id]

                else: # if trilateration fails
                    # remove the first observation, and let the robot makes a new observation later
                    self.feature_observation_ranges[id].pop(0)
                    self.feature_observation_points[id].pop(0)

        # Debugging
        print("Features in states = ", self.feature_states)
        print("Features in states id = ", self.feature_states_id)



    ## Other functions ###################

    def encoder_to_displacement(self,wl,wr,Re):
        # set time interval
        # previous time = the later timestamp between the 2 wheels
        ti_1 = max(self.left_wheel_prev_time,self.right_wheel_prev_time)

        # current time = now
        ti = time.time()

        # time step
        dt = ti - ti_1

        # convert angular to linear velocity
        vl = wl * self.wheel_radius
        vr = wr * self.wheel_radius

        # displacement of each wheel within the time interval
        dl = vl*dt
        dr = vr*dt

        # linear and angular displacement of robot
        d = (dl+dr)/2
        dtheta = (dl-dr)/self.wheelbase
        uk = np.array([d,0,dtheta]).reshape(3,1)

        # converting covariance from pulses to displacement
        # A matrix in the magic table entry
        A = np.array([[0.5,0.5],[0,0],[1/self.wheelbase, -1/self.wheelbase]])
        A = A @ np.diag([dt,dt]) @ np.diag([self.wheel_radius,self.wheel_radius]) 

        Qk = A @ Re @ A.T

        return uk,Qk
    
    def send_messages(self,event):
        # rospy.loginfo("Publishing odometry.")
        # publish Odometry
        self.send_odom()
        # publish tf
        br = tf.TransformBroadcaster()
        br.sendTransform((self.x, self.y, 0),
                        tf.transformations.quaternion_from_euler(0, 0, self.yaw),
                        rospy.Time.now(),
                        "turtlebot/kobuki/base_footprint",
                        "world_ned")

    def send_odom(self):
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

        Pk_expanded = np.array([[self.Pk[0,0],self.Pk[0,1],0,0,0,self.Pk[0,2]],
                                [self.Pk[1,0],self.Pk[1,1],0,0,0,self.Pk[1,2]],
                                [0,0,0,0,0,0],
                                [0,0,0,0,0,0],
                                [0,0,0,0,0,0],
                                [self.Pk[2,0],self.Pk[2,1],0,0,0,self.Pk[2,2]]])
        odom_msg.pose.covariance = list(Pk_expanded.flatten())

        odom_msg.header.frame_id = "world_ned"

        odom_msg.child_frame_id = "turtlebot/kobuki/base_footprint"

        odom_msg.header.stamp = rospy.Time.now()

        self.odom_pub.publish(odom_msg)

    def trilateration(self,observation_ranges,observation_points):
        # extract coordinates of observation points
        x1, y1 = observation_points[0]
        x2, y2 = observation_points[1]
        x3, y3 = observation_points[2]

        # Extract distances
        d1, d2, d3 = observation_ranges
        
        # Calculate coefficients for linear system
        A = 2 * np.array([
            [x3 - x1, y3 - y1],
            [x3 - x2, y3 - y2]
        ])
        
        b = np.array([
            (d1 ** 2 - d3 ** 2 + x3 ** 2 - x1 ** 2 + y3 ** 2 - y1 ** 2),
            (d2 ** 2 - d3 ** 2 + x3 ** 2 - x2 ** 2 + y3 ** 2 - y2 ** 2)
        ])
        
        # Solve linear system
        try:
            xf, yf = np.linalg.solve(A, b)
            return xf, yf
        except np.linalg.LinAlgError:
            # If the linear system is singular (points are collinear), return None
            return None, None



if __name__=='__main__':
    rospy.init_node('turtlebot_localization', anonymous=True) # initialize the node
    node = TurtlebotLocalization() # a newly-created class

    rospy.spin()
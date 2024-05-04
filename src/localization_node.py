#!/usr/bin/env python3

'''
    Node to perform localization using EKF.
'''

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu
import tf.transformations
from visualization_msgs.msg import Marker,MarkerArray
import tf
import time

from filter import *
from ho_localization_project.msg import ArucoRange # a newly created ROS msg

from geometry_msgs.msg import PointStamped

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

        # transformation from base_footprint to camera frame (for trilateration)
        self.base_to_cam = np.zeros((3,1))

        # instantiate filter
        self.filter = EKF(np.array([self.x,self.y,self.yaw]).reshape(3,1),self.Pk)

        # Subscribers
        self.encoder_sub = rospy.Subscriber("/turtlebot/joint_states",JointState,self.encoder_callback)
        self.imu_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/imu_data",Imu,self.imu_callback)
        self.feature_sub = rospy.Subscriber("/aruco_range",ArucoRange,self.feature_callback)

        # Publishers
        self.odom_pub = rospy.Publisher("/odom",Odometry, queue_size=1)
        self.marker_pub = rospy.Publisher("~aruco_marker", MarkerArray, queue_size=1)

        self.obs_point_pub = rospy.Publisher("/obs_point_vis", PointStamped, queue_size=1)
        self.aruco_ranges_pub = rospy.Publisher("/aruco_range_vis", MarkerArray, queue_size=1)
        self.trilateration_pub = rospy.Publisher("/trilateration_vis", MarkerArray, queue_size=1)

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
        # obtain tf from base_footprint to camera frame
        if not self.base_to_cam.any():
            try:
                listener = tf.TransformListener()
                (trans,rot) = listener.lookupTransform('turtlebot/kobuki/base_footprint', 'camera_color_frame', rospy.Time(0))
                _,_,angle = tf.transformations.euler_from_quaternion(rot)
                self.base_to_cam = np.array([trans[0],trans[1],angle]).reshape((3,1))
            except:
                pass
        
        # get observation point = tf from world_ned to camera_color_frame
        # obtained from pose compounding of odom \oplus (base_footprint to camera)
        world_to_base = np.array([self.x,self.y,self.yaw]).reshape((3,1))
        obs_point = Pose3D(world_to_base).oplus(self.base_to_cam)

        print("Range msg = ",range_msg)
        marker_id = list(range_msg.id)
        marker_range = list(range_msg.range)

        # Debugging: publish Aruco ranges to Rviz
        self.publish_aruco_ranges_vis(marker_range,obs_point)

        print("Marker id = ", marker_id)
        print("Marker range = ", marker_range)

        for n in range(len(marker_id)):
            id = marker_id[n]
            # only add features to the states if it still doesn't exist in the states
            if id not in self.feature_states_id:
                              
                # Debugging: publish obs_point to Rviz
                self.publish_obs_point(obs_point)
                
                # store the ranges and observation points for trilateration
                if id not in self.feature_observation_ranges.keys(): # marker ID observed for the first time
                    self.feature_observation_ranges[id] = [marker_range[n]]
                    self.feature_observation_points[id] = [(obs_point[0,0],obs_point[1,0])]
                else:
                    self.feature_observation_ranges[id].append(marker_range[n])
                    self.feature_observation_points[id].append((obs_point[0,0],obs_point[1,0]))

        print("Feature observation ranges = ", self.feature_observation_ranges)
        print("Feature observation points = ", self.feature_observation_points)
        
        # Perform trilateration
        feature_observation_ranges_copy = self.feature_observation_ranges.copy() # to avoid changing length of dict in iteration
        for id in feature_observation_ranges_copy.keys():
            # check if the marker already has 3 observations
            if len(self.feature_observation_ranges[id]) == 3:
                print(f"Perform trilateration on feature {id}.")
                xfi,Pfi = self.trilateration(self.feature_observation_ranges[id],self.feature_observation_points[id])
                xf,yf = xfi[0,0],xfi[1,0]
                # if trilateration succeeds, add the feature to the states
                if xf and yf:
                    # Debugging: visualize the 3 ranges in Rviz
                    self.publish_trilateration_ranges(self.feature_observation_ranges[id],self.feature_observation_points[id],xf,yf)
                    print("Trilateration succeed.")
                    self.feature_states.append([xf,yf])
                    self.feature_states_id.append(id)
                    
                    # remove it from the list of observations to trilaterate
                    del self.feature_observation_ranges[id]
                    del self.feature_observation_points[id]

                else: # if trilateration fails
                    print("Trilateration failed.")
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
        
        # publish feature visualization
        self.send_feature()

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
        # don't perform trilateration if the distance between observation points is less than a threshold
        dist_threshold = 0.2
        dist1 = np.linalg.norm(np.subtract(observation_points[0],observation_points[1]))
        dist2 = np.linalg.norm(np.subtract(observation_points[1],observation_points[2]))
        if (dist1 < dist_threshold) or (dist2 < dist_threshold):
            xk = np.array([[None],[None]])
            return xk, None

        # extract coordinates of observation points
        p1 = np.array(observation_points[0]).reshape((-1,1))
        p2 = np.array(observation_points[1]).reshape((-1,1))
        p3 = np.array(observation_points[2]).reshape((-1,1))

        P = np.block([p1,p2,p3])

        # Extract distances
        r = np.array(observation_ranges).T

        # Hard-code range uncertainties (?)
        R = np.diag([0.2,0.2,0.2])

        # Unconstrained least squares multilateration formulation
        d = r * r # Hadamard product
        H = np.block([2*P.T, -1*np.ones((3,1))])
        z = (np.diag(P.T @ P)).reshape((-1,1)) - d
        Pxi = 4 * np.diag(r.flatten()) @ R @ np.diag(r.flatten())
        W = np.linalg.inv(Pxi)
        theta_WLS = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ z
        N = np.block([np.eye(2),np.zeros((2,1))])
        xk = N @ theta_WLS
        Pk = N @ np.linalg.inv(H.T @ W @ H) @ N.T

        return xk,Pk
        
    def send_feature(self):
        markers_list = []
        # create Marker message for each feature in state
        for feature in range(len(self.feature_states_id)):
            m = Marker()
            m.header.frame_id = "world_ned"
            m.header.stamp = rospy.Time.now()
            m.ns = "feature"
            m.id = self.feature_states_id[feature]
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = self.feature_states[feature][0]
            m.pose.position.y = self.feature_states[feature][1]
            m.pose.position.z = 0.0

            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1

            m.color.a = 1.0
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0

            markers_list.append(m)

        # publish MarkerArray
        marker_msg = MarkerArray()
        marker_msg.markers = markers_list
        self.marker_pub.publish(marker_msg)

    ## DEBUGGING FUNCTIONS

    def publish_obs_point(self,obs_point):
        msg = PointStamped()
        msg.header.frame_id = "world_ned"
        msg.header.stamp = rospy.Time.now()

        msg.point.x = obs_point[0,0]
        msg.point.y = obs_point[1,0]
        msg.point.z = -0.0945
        self.obs_point_pub.publish(msg)

    def publish_aruco_ranges_vis(self,aruco_ranges,obs_point):
        ranges_list = []
        # create Marker message for each feature in state
        for marker in range(len(aruco_ranges)):
            m = Marker()
            m.header.frame_id = "world_ned"
            m.header.stamp = rospy.Time.now()
            m.ns = f"aruco_ranges_{marker}"
            m.id = 0
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            m.pose.position.x = obs_point[0,0]
            m.pose.position.y = obs_point[1,0]
            m.pose.position.z = 0.0

            m.scale.x = 2*aruco_ranges[marker]
            m.scale.y = 2*aruco_ranges[marker]
            m.scale.z = 0.05

            m.color.a = 0.5
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0

            ranges_list.append(m)

        # publish MarkerArray
        range_vis_msg = MarkerArray()
        range_vis_msg.markers = ranges_list
        self.aruco_ranges_pub.publish(range_vis_msg)

    def publish_trilateration_ranges(self,ranges,obs_points,xf,yf):
        obs_list = []
        # create Marker message for each feature in state
        for obs in range(len(ranges)):
            # show the ranges
            m1 = Marker()
            m1.header.frame_id = "world_ned"
            m1.header.stamp = rospy.Time.now()
            m1.ns = f"observation_{obs}"
            m1.id = 0
            m1.type = Marker.CYLINDER
            m1.action = Marker.ADD

            m1.pose.position.x = obs_points[obs][0]
            m1.pose.position.y = obs_points[obs][1]
            m1.pose.position.z = 0.0

            m1.scale.x = 2*ranges[obs]
            m1.scale.y = 2*ranges[obs]
            m1.scale.z = 0.05

            m1.color.a = 0.5
            m1.color.r = 0.3*(obs+1)
            m1.color.g = 0.0
            m1.color.b = 0.3*(3-obs)

            obs_list.append(m1)

            # show the observation points
            m2 = Marker()
            m2.header.frame_id = "world_ned"
            m2.header.stamp = rospy.Time.now()
            m2.ns = f"observation_{obs}"
            m2.id = 1
            m2.type = Marker.SPHERE
            m2.action = Marker.ADD

            m2.pose.position.x = obs_points[obs][0]
            m2.pose.position.y = obs_points[obs][1]
            m2.pose.position.z = 0.0

            m2.scale.x = 0.1
            m2.scale.y = 0.1
            m2.scale.z = 0.1

            m2.color.a = 1.0
            m2.color.r = 0.3*(obs+1)
            m2.color.g = 0.0
            m2.color.b = 0.3*(3-obs)

            obs_list.append(m2)

        # show the trilateration result
        m3 = Marker()
        m3.header.frame_id = "world_ned"
        m3.header.stamp = rospy.Time.now()
        m3.ns = "result"
        m3.id = 0
        m3.type = Marker.SPHERE
        m3.action = Marker.ADD

        m3.pose.position.x = xf
        m3.pose.position.y = yf
        m3.pose.position.z = 0.0

        m3.scale.x = 0.2
        m3.scale.y = 0.2
        m3.scale.z = 0.2

        m3.color.a = 1.0
        m3.color.r = 0.0
        m3.color.g = 1.0
        m3.color.b = 0.0

        obs_list.append(m3)

        # publish MarkerArray
        obs_vis_msg = MarkerArray()
        obs_vis_msg.markers = obs_list
        self.trilateration_pub.publish(obs_vis_msg)




if __name__=='__main__':
    rospy.init_node('turtlebot_localization', anonymous=True) # initialize the node
    node = TurtlebotLocalization() # a newly-created class

    rospy.spin()
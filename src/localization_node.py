#!/usr/bin/env python3

'''
    Node to perform localization using EKF.
'''

import scipy.linalg
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu
import tf.transformations
from visualization_msgs.msg import Marker,MarkerArray
import tf

import scipy

from utils.filter import *
from ho_localization_project.msg import ArucoRange # a newly created ROS msg
from utils.GetEllipse import GetEllipse

from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import ColorRGBA

class TurtlebotLocalization:
    def __init__(self) -> None:

        # initialize state vector with the initial robot pose
        self.xk = np.zeros((3,1))
        # self.xk = np.array([3.0,-0.78,0.0]).reshape((3,1)) # to be used with hol_circuit2 scenario

        # initialize robot pose covariance
        self.Pk = np.zeros((3,3))

        # initialize wheel angular velocities
        self.wl = 0 # left wheel
        self.wr = 0 # right wheel

        # covariances
        # TODO: tune here!
        self.cov_encoder = 0.1
        self.cov_imu = 0.001
        self.cov_feature_init = 0.5
        self.cov_feature_update = 0.3

        # initialize encoder covariance
        self.left_wheel_cov = 0.0 # left wheel
        self.right_wheel_cov = 0.0 # right wheel
        self.Re = np.diag([self.left_wheel_cov,self.right_wheel_cov])

        # initialize imu data
        self.orientation = 0 # from magnetometer

        # flag attribute to indicate start of localization
        self.start = False

        # encoder timestamp
        self.left_wheel_time = 0
        self.right_wheel_time = 0

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
        self.feature_observation_ranges = {} # dict of ranges for multilateration
        self.feature_observation_points = {} # dict of observation points for multilateration

        # transformation from base_footprint to camera frame (for multilateration)
        self.base_to_cam = np.zeros((3,1))

        # instantiate filter
        self.filter = EKF(self.xk,self.Pk)

        # Subscribers
        self.encoder_sub = rospy.Subscriber("/turtlebot/joint_states",JointState,self.encoder_callback)
        self.imu_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/imu_data",Imu,self.imu_callback)
        self.feature_sub = rospy.Subscriber("/aruco_range",ArucoRange,self.feature_callback)

        # Publishers
        self.odom_pub = rospy.Publisher("/odom",Odometry, queue_size=1)
        self.marker_pub = rospy.Publisher("~aruco_marker", MarkerArray, queue_size=1)
        self.multilateration_pub = rospy.Publisher("/trilateration_vis", MarkerArray, queue_size=1)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(0.1), self.send_messages)


    ## DEAD-RECKONING ###################

    def encoder_callback(self,states_msg):
        if self.start:
            # assign the wheel encoder values
            names_list = list(states_msg.name)
            if "turtlebot/kobuki/wheel_left_joint" in names_list:
                left_index = names_list.index("turtlebot/kobuki/wheel_left_joint")
                # set left wheel
                self.wl = states_msg.velocity[left_index]
                self.left_wheel_cov = self.cov_encoder
                self.left_wheel_flag = True
                self.left_wheel_prev_time = self.left_wheel_time
                self.left_wheel_time = states_msg.header.stamp

            if "turtlebot/kobuki/wheel_right_joint" in names_list:
                right_index = names_list.index("turtlebot/kobuki/wheel_right_joint")
                # set right wheel
                self.wr = states_msg.velocity[right_index]
                self.right_wheel_cov = self.cov_encoder
                self.right_wheel_flag = True
                self.right_wheel_prev_time = self.right_wheel_time
                self.right_wheel_time = states_msg.header.stamp
            
            if self.right_wheel_flag and self.left_wheel_flag:
                # Pre-processing for Prediction
                xk_1 = self.xk
                Pk_1 = self.Pk
                self.Re = np.diag([self.left_wheel_cov,self.right_wheel_cov])

                # Convert encoder reading to odometry displacement
                uk,Qk = self.encoder_to_displacement(self.wl,self.wr,self.Re)

                # Perform Prediction
                xk_bar,Pk_bar = self.filter.Prediction(xk_1,Pk_1,uk,Qk)

                # Save Prediction result
                self.xk = xk_bar
                self.Pk = Pk_bar

                # Reset the flags
                self.right_wheel_flag = False
                self.left_wheel_flag = False

                # Save current time
                self.current_time = states_msg.header.stamp

    def imu_callback(self,imu_msg):        
        # Get the yaw reading from the magnetometer
        _,_,yaw = tf.transformations.euler_from_quaternion([imu_msg.orientation.x, 
                                                            imu_msg.orientation.y,
                                                            imu_msg.orientation.z,
                                                            imu_msg.orientation.w])
        
        # Start the localization when the node receives the first IMU reading
        # NOTE: remove the negative sign for yaw if testing in simulation
        if not self.start:
            self.xk[2,0] = -yaw # store the first IMU yaw reading as the initial yaw value
            self.start = True # start the localization
            # store also initial time
            self.left_wheel_time = imu_msg.header.stamp
            self.right_wheel_time = imu_msg.header.stamp
            self.current_time = imu_msg.header.stamp
        else:
            # Pre-processing for Update
            zk = np.array([[-yaw]])
            # Rk = np.array([[imu_msg.orientation_covariance[8]]])
            Rk = np.array([[self.cov_imu]]) # covariance of IMU hard-coded
            Hk = np.array([[0,0,1]]) # for robot pose only
            Hk = np.block([Hk,np.zeros((1,self.xk.shape[0]-3))])
            Vk = np.eye(1)

            xk_bar = self.xk
            Pk_bar = self.Pk

            # Perform Update
            xk,Pk = self.filter.UpdateMeasurement(xk_bar,Pk_bar,zk,Rk,Hk,Vk)

            # Save Update results
            self.xk = xk
            self.Pk = Pk

            # Save current time
            self.current_time = imu_msg.header.stamp

    def encoder_to_displacement(self,wl,wr,Re):
        # set time interval
        # previous time = the later timestamp between the 2 wheels
        ti_1 = max(self.left_wheel_prev_time,self.right_wheel_prev_time)

        # current time = now
        ti = max(self.left_wheel_time,self.right_wheel_time)

        # time step
        dt = (ti - ti_1).to_sec()

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

    ## FEATURE OBSERVATION ########################

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
        world_to_base = self.xk[:3]
        obs_point = Pose3D(world_to_base).oplus(self.base_to_cam)

        # Debugging: publish obs_point to Rviz
        self.publish_obs_point(obs_point)

        # Extract observed markers
        marker_id = list(range_msg.id)
        marker_range = list(range_msg.range)

        # Debugging: publish Aruco ranges to Rviz
        self.publish_aruco_ranges_vis(marker_range,obs_point)

        # Initialize observation vector and hypothesis list for feature update
        zf = np.zeros((0,1))
        Rf = np.zeros((0,0))
        zf_ids = []

        ## Split observed markers between mapped and unmapped
        for n in range(len(marker_id)): # loop for all markers observed
            id_n = marker_id[n]
            range_n = marker_range[n]

            if id_n in self.feature_states_id: # if marker is already in the state vector
                # store marker for update
                zf = np.vstack((zf,np.array([[range_n]])))
                Rf = scipy.linalg.block_diag(Rf,np.array([[self.cov_feature_update]])) # range measurement uncertainty hard-coded
                zf_ids.append(id_n)
            else:                             
                # store the ranges and observation points for multilateration
                self.prepare_feature_initialization(id_n,range_n,obs_point)
        
        ## Perform update on mapped features
        if zf.any():
            # Pre-processing for update
            xk_bar = self.xk
            Pk_bar = self.Pk
            H = self.filter.DataAssociation(zf_ids, self.feature_states_id)
            Hk = self.filter.Jhfx(xk_bar,H)
            Vk = np.eye(len(H))

            xk,Pk = self.filter.UpdateFeature(xk_bar,Pk_bar,zf,Rf,Hk,Vk,H)

            # Save Update results
            self.xk = xk
            self.Pk = Pk

        ## Perform state augmentation on unmapped features
        self.state_augmentation()


    def prepare_feature_initialization(self, id, marker_range, obs_point):
        # store the ranges and observation points for multilateration
        if id not in self.feature_observation_ranges.keys(): # marker ID observed for the first time
            self.feature_observation_ranges[id] = [marker_range]
            self.feature_observation_points[id] = [(obs_point[0,0],obs_point[1,0])]
        else:
            self.feature_observation_ranges[id].append(marker_range)
            self.feature_observation_points[id].append((obs_point[0,0],obs_point[1,0]))


    def state_augmentation(self):
        feature_observation_ranges_copy = self.feature_observation_ranges.copy() # to avoid changing length of dict in iteration
        for id in feature_observation_ranges_copy.keys():
            # check if the marker already has 4 observations
            if len(self.feature_observation_ranges[id]) == 4:
                print(f"Perform multilateration on feature {id}.")
                xfi,Pfi = self.multilateration(self.feature_observation_ranges[id],self.feature_observation_points[id])
                xf,yf = xfi[0,0],xfi[1,0]
                # if multilateration succeeds, add the feature to the states
                if xf and yf:
                    # Debugging: visualize the ranges in Rviz
                    self.publish_multilateration_ranges(self.feature_observation_ranges[id],self.feature_observation_points[id],xf,yf)
                    print("Multilateration succeed.")
                    self.feature_states.append([xf,yf])
                    self.xk = np.vstack((self.xk,xfi)) # append to the state vector
                    self.feature_states_id.append(id)
                    self.Pk = scipy.linalg.block_diag(self.Pk,Pfi)
                    
                    # remove it from the list of observations to trilaterate
                    del self.feature_observation_ranges[id]
                    del self.feature_observation_points[id]

                else: # if multilateration fails
                    print("Multilateration failed.")
                    # remove the first observation, and let the robot makes a new observation later
                    self.feature_observation_ranges[id].pop(0)
                    self.feature_observation_points[id].pop(0)


    def multilateration(self,observation_ranges,observation_points):
        # don't perform multilateration if the distance between observation points is less than a threshold
        dist_threshold = 0.1
        dist1 = np.linalg.norm(np.subtract(observation_points[0],observation_points[1]))
        dist2 = np.linalg.norm(np.subtract(observation_points[1],observation_points[2]))
        dist3 = np.linalg.norm(np.subtract(observation_points[2],observation_points[3]))
        if (dist1 < dist_threshold) or (dist2 < dist_threshold) or (dist3 < dist_threshold):
            xk = np.array([[None],[None]])
            return xk, None

        # extract coordinates of observation points
        p1 = np.array(observation_points[0]).reshape((-1,1))
        p2 = np.array(observation_points[1]).reshape((-1,1))
        p3 = np.array(observation_points[2]).reshape((-1,1))
        p4 = np.array(observation_points[3]).reshape((-1,1))

        P = np.block([p1,p2,p3,p4])

        # Extract distances
        r = np.array(observation_ranges).reshape((-1,1))

        # Hard-code range uncertainties
        R = np.eye(4) * self.cov_feature_init

        # Unconstrained least squares multilateration formulation
        d = r * r # Hadamard product
        H = np.block([2*P.T, -1*np.ones((P.shape[1],1))])
        z = (np.diag(P.T @ P)).reshape((-1,1)) - d
        Pxi = 4 * np.diag(r.flatten()) @ R @ np.diag(r.flatten())
        W = np.linalg.inv(Pxi)
        theta_WLS = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ z
        N = np.block([np.eye(2),np.zeros((2,1))])
        xk = N @ theta_WLS
        Pk = N @ np.linalg.inv(H.T @ W @ H) @ N.T

        return xk,Pk
    
    ## PUBLISHER FUNCTIONS ###################

    def send_messages(self,event):
        # rospy.loginfo("Publishing odometry.")
        # publish Odometry
        self.send_odom()
        # publish tf
        br = tf.TransformBroadcaster()
        br.sendTransform((self.xk[0,0], self.xk[1,0], 0),
                        tf.transformations.quaternion_from_euler(0, 0, self.xk[2,0]),
                        self.current_time,
                        "turtlebot/kobuki/base_footprint",
                        "world_ned")
        
        # publish feature visualization
        self.send_feature()

    def send_odom(self):
        # publish Odometry message
        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = self.xk[0,0]
        odom_msg.pose.pose.position.y = self.xk[1,0]
        odom_msg.pose.pose.position.z = 0

        yaw_quaternion = tf.transformations.quaternion_from_euler(0,0,self.xk[2,0])
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

        odom_msg.header.stamp = self.current_time

        self.odom_pub.publish(odom_msg)

        
    def send_feature(self):
        nf = int((self.xk.shape[0] - 3)/2) # number of mapped features

        markers_list = []

        # Visualize the mean position of feature
        # create Point for each feature in state
        mean_m = Marker()
        mean_m.header.frame_id = 'world_ned'
        mean_m.header.stamp = self.current_time
        mean_m.id = 0
        mean_m.type = Marker.SPHERE_LIST
        mean_m.ns = 'mean'
        mean_m.action = Marker.DELETE
        mean_m.lifetime = rospy.Duration(0)

        mean_m.action = Marker.ADD
        mean_m.scale.x = 0.1
        mean_m.scale.y = 0.1
        mean_m.scale.z = 0.1

        mean_m.pose.orientation.x = 0
        mean_m.pose.orientation.y = 0
        mean_m.pose.orientation.z = 0
        mean_m.pose.orientation.w = 1
        
        color_white = ColorRGBA()
        color_white.r = 1
        color_white.g = 1
        color_white.b = 1
        color_white.a = 1
        
        for i in range(nf):
            p = Point()
            p.x = self.xk[2*i+3,0]
            p.y = self.xk[2*i+4,0]
            p.z = 0.0
            mean_m.points.append(p)
            mean_m.colors.append(color_white)

        markers_list.append(mean_m)

        # Visualize covariance
        for i in range(nf):
            # Get uncertainty ellipse points
            f_index = 2*i+3
            xf = self.xk[f_index:f_index+2,0].reshape((2,1))
            Pf = self.Pk[f_index:f_index+2,f_index:f_index+2]
            ellipse_points = GetEllipse(xf,Pf)

            # Create Marker
            cov_m = Marker()
            cov_m.header.frame_id = 'world_ned'
            cov_m.header.stamp = self.current_time
            cov_m.id = 0
            cov_m.type = Marker.LINE_STRIP
            cov_m.ns = f"cov{i}"
            cov_m.action = Marker.DELETE
            cov_m.lifetime = rospy.Duration(0)

            cov_m.action = Marker.ADD
            cov_m.scale.x = 0.02
            cov_m.scale.y = 0
            cov_m.scale.z = 0

            cov_m.pose.orientation.x = 0
            cov_m.pose.orientation.y = 0
            cov_m.pose.orientation.z = 0
            cov_m.pose.orientation.w = 1

            for n in range(ellipse_points.shape[1]):
                p = Point()
                p.x = ellipse_points[0,n]
                p.y = ellipse_points[1,n]
                p.z = 0.0
                cov_m.points.append(p)
                cov_m.colors.append(color_white)

            markers_list.append(cov_m)

        # Label the markers
        for i in range(nf):
            label_m = Marker()
            label_m.header.frame_id = 'world_ned'
            label_m.header.stamp = self.current_time
            label_m.id = 0
            label_m.type = Marker.TEXT_VIEW_FACING
            label_m.ns = f"label{i}"
            label_m.action = Marker.DELETE
            label_m.lifetime = rospy.Duration(0)

            label_m.action = Marker.ADD
            label_m.scale.z = 0.1

            label_m.pose.position.x = self.xk[2*i+3,0] + 0.1
            label_m.pose.position.y = self.xk[2*i+4,0] + 0.1
            label_m.pose.position.z = 0.0
            label_m.pose.orientation.x = 0.0
            label_m.pose.orientation.y = 0.0
            label_m.pose.orientation.z = 0.0
            label_m.pose.orientation.w = 1.0

            label_m.text = "id = " + str(self.feature_states_id[i])

            label_m.color.a = 1.0
            label_m.color.r = 1.0
            label_m.color.g = 1.0
            label_m.color.b = 0.0

            markers_list.append(label_m)

        # publish MarkerArray
        marker_msg = MarkerArray()
        marker_msg.markers = markers_list
        self.marker_pub.publish(marker_msg)

    def publish_multilateration_ranges(self,ranges,obs_points,xf,yf):
        obs_list = []
        # create Marker message for each feature in state
        for obs in range(len(ranges)):
            # show the ranges
            m1 = Marker()
            m1.header.frame_id = "world_ned"
            m1.header.stamp = self.current_time
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
            m1.color.r = 0.25*(obs+1)
            m1.color.g = 0.0
            m1.color.b = 0.25*(4-obs)

            m1.lifetime = rospy.Duration(3.0)

            obs_list.append(m1)

            # show the observation points
            m2 = Marker()
            m2.header.frame_id = "world_ned"
            m2.header.stamp = self.current_time
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
            m2.color.r = 0.25*(obs+1)
            m2.color.g = 0.0
            m2.color.b = 0.25*(4-obs)

            m2.lifetime = rospy.Duration(3.0)

            obs_list.append(m2)

        # show the multilateration result
        m3 = Marker()
        m3.header.frame_id = "world_ned"
        m3.header.stamp = self.current_time
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

        m3.lifetime = rospy.Duration(3.0)

        obs_list.append(m3)

        # publish MarkerArray
        obs_vis_msg = MarkerArray()
        obs_vis_msg.markers = obs_list
        self.multilateration_pub.publish(obs_vis_msg)




if __name__=='__main__':
    rospy.init_node('turtlebot_localization', anonymous=True) # initialize the node
    node = TurtlebotLocalization() # a newly-created class

    rospy.spin()

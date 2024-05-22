import numpy as np
# from math import cos,sin
from utils.Pose import Pose3D

'''
    Pure Python implementation of the Extended Kalman Filter (EKF) for localization,
    using wheel encoders for Prediction, and IMU magnetometer yaw reading for Update.

    Note: Most functions are copied from PR lab.
'''

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

class EKF:
    def __init__(self,x0,P0) -> None:
        self.xk = x0
        self.Pk = P0

    def Prediction(self,xk_1,Pk_1,uk,Qk):
        # self.xk_bar = self.f(xk_1,uk)
        # Ak = self.Jfx(xk_1,uk)
        # Wk = self.Jfw(xk_1)
        # self.Pk_bar = Ak@Pk_1@Ak.T + Wk@Qk@Wk.T

        # self.xk_bar[2,0] = wrap_angle(self.xk_bar[2,0])

        # return self.xk_bar, self.Pk_bar

        ##################################

        # Updated for FEKFSLAM
    
        xk_bar = np.block([self.f(xk_1[:3],uk).T, xk_1[3:max(xk_1.shape)].T]).T

        # split Pk_bar by its elements
        Pk1 = self.Jfx(xk_1[:3],uk) @ Pk_1[:3,:3] @ self.Jfx(xk_1[:3],uk).T + self.Jfw(xk_1[:3]) @ Qk @ self.Jfw(xk_1[:3]).T
        Pk2 = self.Jfx(xk_1[:3],uk) @ Pk_1[:3,3:Pk_1.shape[1]]
        Pk3 = Pk2.T
        Pk4 = Pk_1[3:Pk_1.shape[1],3:Pk_1.shape[1]]

        Pk_bar = np.block([[Pk1,Pk2],[Pk3,Pk4]])

        # wrap the yaw angle
        xk_bar[2,0] = wrap_angle(xk_bar[2,0])

        return xk_bar, Pk_bar

    def UpdateMeasurement(self,xk_bar,Pk_bar,zk,Rk,Hk,Vk):

        Kk = Pk_bar @ Hk.T @ np.linalg.inv(Hk@Pk_bar@Hk.T + Vk@Rk@Vk.T)
        innovation = zk-self.hm(xk_bar)
        innovation[0,0] = wrap_angle(innovation[0,0])
        self.xk = xk_bar + Kk@(innovation)

        B = np.identity(np.shape(Kk@Hk)[0]) - Kk@Hk
        self.Pk = B@Pk_bar@B.T

        self.xk[2,0] = wrap_angle(self.xk[2,0])

        return self.xk,self.Pk
    
    def DataAssociation(self,obs_id_list, state_id_list):
        '''
            Convert ArUco marker IDs into indices.
        '''
        H = []

        for id in obs_id_list:
            index = state_id_list.index(id)
            H.append(index)

        return H
    
    def UpdateFeature(self,xk_bar,Pk_bar,zk,Rk,Hk,Vk,H):

        # print("xk_bar = ", xk_bar)
        # print("Pk_bar = ", Pk_bar)
        # print("zk = ", zk)
        # print("Rk = ", Rk)
        # print("Hk = ", Hk)
        # print("Vk = ", Vk)

        Kk = Pk_bar @ Hk.T @ np.linalg.inv(Hk@Pk_bar@Hk.T + Vk@Rk@Vk.T)
        innovation = zk-self.hf(xk_bar,H)
        self.xk = xk_bar + Kk@(innovation)

        B = np.identity(np.shape(Kk@Hk)[0]) - Kk@Hk
        self.Pk = B@Pk_bar@B.T

        self.xk[2,0] = wrap_angle(self.xk[2,0])

        return self.xk,self.Pk

    def f(self,xk_1,uk):
        xk_bar = Pose3D(xk_1).oplus(uk)

        return xk_bar

    def Jfx(self,xk_1,uk):
        J = Pose3D(xk_1).J_1oplus(uk)

        return J

    def Jfw(self,xk_1):
        J = Pose3D(xk_1).J_2oplus()

        return J
    
    def hm(self, xk):  # return the expected observations
        z = xk[2]

        return z  
    
    def hf(self,xk,H):
        # initialize vector of expected feature observation
        h = np.zeros((0, 1))  # empty vector
        nf = len(H) # number of feature

        for i in range(nf):
            fi = H[i] # H = hypothesis

            if not (fi == None):
                h = np.block([[h], [self.hfi(xk,fi)]])    

        return h

    def hfi(self,xk,i):
        # extract robot position
        xr = xk[0,0]
        yr = xk[1,0]

        # extract feature position at index i
        xfi = xk[2*i + 3]
        yfi = xk[2*i + 4]

        return np.sqrt((xfi-xr)**2 + (yfi-yr)**2)
    
    def Jhfx(self,xk,H):
        # initialize Jacobian
        J = np.zeros((0,xk.shape[0]))
        nf = len(H) # number of feature

        for i in range(nf):
            fi = H[i]

            if not (fi == None):
                J = np.block([[J], [self.Jhfix(xk,fi)]])

        return J
    
    def Jhfix(self,xk,i):
        # extract robot position
        xr = xk[0,0]
        yr = xk[1,0]

        # extract feature position at index i
        xfi = xk[2*i + 3]
        yfi = xk[2*i + 4]

        # initialize row Jacobian
        J = np.zeros((1,xk.shape[0]))

        # fill the elements of J
        J[0,0] = (xr-xfi)/np.sqrt((xfi-xr)**2 + (yfi-yr)**2)
        J[0,1] = (yr-yfi)/np.sqrt((xfi-xr)**2 + (yfi-yr)**2)

        J[0,2*i+3] = (-xr+xfi)/np.sqrt((xfi-xr)**2 + (yfi-yr)**2)
        J[0,2*i+4] = (-yr+yfi)/np.sqrt((xfi-xr)**2 + (yfi-yr)**2)

        return J

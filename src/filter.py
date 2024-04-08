import numpy as np
from math import cos,sin

class EKF:
    def __init__(self,x0,P0) -> None:
        self.xk = x0
        self.Pk = P0

    def Prediction(self,xk_1,Pk_1,uk,Qk):
        self.xk_bar = self.f(xk_1,uk)
        Ak = self.Jfx(xk_1,uk)
        Wk = self.Jfw(xk_1)
        self.Pk_bar = Ak@Pk_1@Ak.T + Wk@Qk@Wk.T

        return self.xk_bar, self.Pk_bar

    def Update(self,xk_bar,Pk_bar,zk,Rk,Hk,Vk):
        # print("Pk_bar = ",Pk_bar.shape)
        # print("Hk = ", Hk.shape)
        # print("Vk = ", Vk.shape)
        # print("Rk = ", Rk.shape)

        Kk = Pk_bar @ Hk.T @ np.linalg.inv(Hk@Pk_bar@Hk.T + Vk@Rk@Vk.T)
        self.xk = xk_bar + Kk@(zk-self.h(xk_bar))

        B = np.identity(np.shape(Kk@Hk)[0]) - Kk@Hk
        self.Pk = B@Pk_bar@B.T

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
    
    def h(self, xk):  # return the expected observations
        z = xk[2]

        return z  
    

class Pose3D:
    def __new__(cls, input_array=np.array([[0.0, 0.0, 0.0]]).T):
        """
        Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarry numpy class.

        :param input_array: array used to initialize the class
        :returns: the instance of a Pose4D class object
        """

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj
    
    def __init__(self, input_array=np.array([[0.0, 0.0, 0.0]]).T):

        assert input_array.shape == (3, 1), "mean must be a 3x1 vector"

    def oplus(AxB, BxC):
        """
        Given a Pose3D object *AxB* (the self object) and a Pose3D object *BxC*, it returns the Pose3D object *AxC*.

        .. math::
            \\mathbf{{^A}x_B} &= \\begin{bmatrix} ^Ax_B & ^Ay_B & ^A\\psi_B \\end{bmatrix}^T \\\\
            \\mathbf{{^B}x_C} &= \\begin{bmatrix} ^Bx_C & ^By_C & & ^B\\psi_C \\end{bmatrix}^T \\\\

        The operation is defined as:

        .. math::
            \\mathbf{{^A}x_C} &= \\mathbf{{^A}x_B} \\oplus \\mathbf{{^B}x_C} =
            \\begin{bmatrix}
                ^Ax_B + ^Bx_C  \\cos(^A\\psi_B) - ^By_C  \\sin(^A\\psi_B) \\\\
                ^Ay_B + ^Bx_C  \\sin(^A\\psi_B) + ^By_C  \\cos(^A\\psi_B) \\\\
                ^A\\psi_B + ^B\\psi_C
            \\end{bmatrix}
            :label: eq-oplus3dof

        :param BxC: C-Frame pose expressed in B-Frame coordinates
        :returns: C-Frame pose expressed in A-Frame coordinates
        """
        
        # define AxC
        AxC = np.empty((3,1))
        
        # compute the direct compounding
        AxC[0] = AxB[0] + BxC[0]*cos(AxB[2]) - BxC[1]*sin(AxB[2])
        AxC[1] = AxB[1] + BxC[0]*sin(AxB[2]) + BxC[1]*cos(AxB[2])
        AxC[2] = AxB[2] + BxC[2]

        return AxC

    def J_1oplus(AxB, BxC):
        """
        Jacobian of the pose compounding operation (eq. :eq:`eq-oplus3dof`) with respect to the first pose:

        .. math::
            J_{1\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Ax_B} =
            \\begin{bmatrix}
                1 & 0 &  -^Bx_C \\sin(^A\\psi_B) - ^By_C \\cos(^A\\psi_B) \\\\
                0 & 1 &  ^Bx_C \\cos(^A\\psi_B) - ^By_C \\sin(^A\\psi_B) \\\\
                0 & 0 & 1
            \\end{bmatrix}
            :label: eq-J1oplus3dof

        The method returns a numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object) and the :math:`2^{nd}` posepose *BxC*.

        :param BxC: 2nd pose
        :returns: Evaluation of the :math:`J_{1\\oplus}` Jacobian of the pose compounding operation with respect to the first pose (eq. :eq:`eq-J1oplus3dof`)
        """

        J1 = np.array([[1,0,-BxC[0,0]*sin(AxB[2,0])-BxC[1,0]*cos(AxB[2,0])],
                       [0,1,BxC[0,0]*cos(AxB[2,0])-BxC[1,0]*sin(AxB[2,0])],
                       [0,0,1]])

        return J1

    def J_2oplus(AxB):
        """
        Jacobian of the pose compounding operation (:eq:`eq-oplus3dof`) with respect to the second pose:

        .. math::
            J_{2\\oplus}=\\frac{\\partial  ^Ax_B \\oplus ^Bx_C}{\\partial ^Bx_C} =
            \\begin{bmatrix}
                \\cos(^A\\psi_B) & -\\sin(^A\\psi_B) & 0  \\\\
                \\sin(^A\\psi_B) & \\cos(^A\\psi_B) & 0  \\\\
                0 & 0 & 1
            \\end{bmatrix}
            :label: eq-J2oplus3dof

        The method returns a numerical matrix containing the evaluation of the Jacobian for the :math:`1^{st} posepose *AxB* (the self object).

        :returns: Evaluation of the :math:`J_{2\\oplus}` Jacobian of the pose compounding operation with respect to the second pose (eq. :eq:`eq-J2oplus3dof`)
        """

        J2 = np.array([[cos(AxB[2,0]), -sin(AxB[2,0]), 0],
                       [sin(AxB[2,0]), cos(AxB[2,0]), 0],
                       [0,0,1]])

        return J2

    def ominus(AxB):
        """
        Inverse pose compounding of the *AxB* pose (the self objetc):

        .. math::
            ^Bx_A = \\ominus ^Ax_B =
            \\begin{bmatrix}
                -^Ax_B \\cos(^A\\psi_B) - ^Ay_B \\sin(^A\\psi_B) \\\\
                ^Ax_B \\sin(^A\\psi_B) - ^Ay_B \\cos(^A\\psi_B) \\\\
                -^A\\psi_B
            \\end{bmatrix}
            :label: eq-ominus3dof

        :returns: A-Frame pose expressed in B-Frame coordinates (eq. :eq:`eq-ominus3dof`)
        """

        # define BxA
        BxA = np.empty((3,1))

        # compute inverse compounding
        BxA[0] = -AxB[0]*cos(AxB[2]) - AxB[1]*sin(AxB[2])
        BxA[1] = AxB[0]*sin(AxB[2]) - AxB[1]*cos(AxB[2])
        BxA[2] = -AxB[2]

        return Pose3D(BxA)

    def J_ominus(AxB):
        """
        Jacobian of the inverse pose compounding operation (:eq:`eq-oplus3dof`) with respect the pose *AxB* (the self object):

        .. math::
            J_{\\ominus}=\\frac{\\partial  \\ominus ^Ax_B}{\\partial ^Ax_B} =
            \\begin{bmatrix}
                -\\cos(^A\\psi_B) & -\\sin(^A\\psi_B) &  ^Ax_B \\sin(^A\\psi_B) - ^Ay_B \\cos(^A\\psi_B) \\\\
                \\sin(^A\\psi_B) & -\\cos(^A\\psi_B) &  ^Ax_B \\cos(^A\\psi_B) + ^Ay_B \\sin(^A\\psi_B) \\\\
                0 & 0 & -1
            \\end{bmatrix}
            :label: eq-Jominus3dof

        Returns the numerical matrix containing the evaluation of the Jacobian for the pose *AxB* (the self object).

        :returns: Evaluation of the :math:`J_{\\ominus}` Jacobian of the inverse pose compounding operation with respect to the pose (eq. :eq:`eq-Jominus3dof`)
        """

        J = np.array([[-cos(AxB[2,0]), -sin(AxB[2,0]), AxB[0,0]*sin(AxB[2,0])-AxB[1,0]*cos(AxB[2,0])],
                      [sin(AxB[2,0]), -cos(AxB[2,0]), AxB[0,0]*cos(AxB[2,0])+AxB[1,0]*sin(AxB[2,0])],
                      [0,0,-1]])

        return J


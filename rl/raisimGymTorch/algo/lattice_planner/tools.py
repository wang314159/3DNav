import math
from typing import List, Tuple
import numpy as np
from scipy import sparse
from math import *
import bisect
import osqp
# from macro_definition import *
import torch


class QuinticPolynomial_torch(object):
    #生成计算横向轨迹点，纵向的停车和跟车的速度
    def __init__(self, xs: float, vxs: float, axs: float, xe: float, vxe: float, axe: float, T: float):#五次多项式
        self.xs = xs#纵向
        self.vxs = vxs
        self.axs = axs
        self.xe = xe#横向
        self.vxe = vxe
        self.axe = axe
        self.T=T
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        
        self.sample_param = T

        A = torch.tensor([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]],dtype=torch.float)
        b =  torch.tensor([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2],dtype=torch.float)
        x = torch.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        """
        return the t state based on QuinticPolynomial theory
        """
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5
        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4
        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2
        return xt


class QuinticPolynomial(object):
    # 生成计算横向轨迹点，纵向的停车和跟车的速度
    def __init__(self, xs: float, vxs: float, axs: float, xe: float, vxe: float, axe: float, T: float):  # 五次多项式
        self.xs = xs  # 纵向
        self.vxs = vxs
        self.axs = axs
        self.xe = xe  # 横向
        self.vxe = vxe
        self.axe = axe
        self.T = T
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        self.sample_param = T

        A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T ** 2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t: float):
        """
        return the t state based on QuinticPolynomial theory
        """
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5
        return xt

    def calc_first_derivative(self, t: float):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4
        return xt

    def calc_second_derivative(self, t: float):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3
        return xt

    def calc_third_derivative(self, t: float):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
        return xt


class QuarticPolynomial(object):  # 四次多项式
    # 生成纵向巡航的速度
    def __init__(self, xs, vxs, axs, vxe, axe, T):
        self.xs = xs  # 初始s状态
        self.vxs = vxs  # 初始速度状态
        self.axs = axs  # 初始加速度状态
        self.vxe = vxe  # 末尾速度状态
        self.axe = axe  # 末尾加速度状态
        self.T = T
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        self.sample_param = T

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                          [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                          axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4
        return xt
    
    def calc_sum(self, t):
        xt = self.a0 * t + 1/2 * self.a1 * t ** 2 + 1/3 * self.a2 * t ** 3 + \
             1/4 * self.a3 * t ** 4 + 1/5 * self.a4 * t ** 5
        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3
        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t
        return xt
class QuarticPolynomial_torch(object):#四次多项式
    #生成纵向巡航的速度
    def __init__(self, xs, vxs, axs, vxe, axe, T):
        self.xs = xs#初始s状态
        self.vxs = vxs#初始速度状态
        self.axs = axs#初始加速度状态
        self.vxe = vxe#末尾速度状态
        self.axe = axe#末尾加速度状态
        self.T=T
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        
        self.sample_param = T

        A = torch.tensor([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]],dtype=torch.float)
        b = torch.tensor([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2],dtype=torch.float)
        #x = np.linalg.solve(A, b)
        x=torch.linalg.solve(A,b)
        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4
        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3
        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t
        return xt

class Spline:#3次线性插值多项式
    """
    Cubic Spline class
    """

    def __init__(self, x: List, y: List):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)

        # with threadpool_limits(limits=2):
        self.c = np.linalg.solve(A, B)

        if isnan(self.c.max()) or isnan(self.c.min()):
            raise RuntimeError("Spline resolve failed")
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t: float) -> float:
        """
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        if i>=len(self.b):
            result=self.y[i]
        else:
            dx = t - self.x[i]
            result = self.a[i] + self.b[i] * dx + \
                self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0


        return result

    def calcd(self, t: float) -> float:
        """
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None


        i = self.__search_index(t)
        if i>=len(self.b):
            result=self.y[-1]-self.y[-2]
        else:
            dx = t - self.x[i]
            result = (self.b[i]
                      + 2.0 * self.c[i]
                      * dx + 3.0 * self.d[i]
                      * dx ** 2.0)
        return result

    def calcdd(self, t: float) -> float:
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        if i>=len(self.d):
            dx = t - self.x[i]
            result = 2.0 * self.c[-1] + 6.0 * self.d[-1] * dx
        else:
            dx = t - self.x[i]
            result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result
    
    def calcddd(self, t: float) -> float:
        """
        Calc third derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        if i >= len(self.d):
            dx = t - self.x[i]
            result = 6.0 * self.d[-1] * dx
        else:
            dx = t - self.x[i]
            result = 6.0 * self.d[i] * dx

        return result

    def __search_index(self, x: float):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1#取比自身的坐标

    def __calc_A(self, h: List[float]):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h: float):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i] 
        return B


class Spline2D:
    """
    2D Cubic Spline class
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)#xy每个点之间的距离
        s = [0]
        s.extend(np.cumsum(self.ds))#元素累加
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)#
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))#曲率
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = atan2(dy, dx)
        return yaw


class PathSmoother:
    def __init__(self) -> None:
        #weight , from config
        self.weight_fem_pos_deviation_ = 1e10 #cost1 - x
        self.weight_path_length = 1          #cost2 - y
        self.weight_ref_deviation = 1        #cost3 - z

    def smooth_path(self,x_array: List, y_array: List) -> Tuple[List, List]:
        length = len(x_array)
        P = np.zeros((length,length))
        #set P matrix,from calculateKernel
        #add cost1
        P[0,0] = 1 * self.weight_fem_pos_deviation_
        P[0,1] = -2 * self.weight_fem_pos_deviation_
        P[1,1] = 5 * self.weight_fem_pos_deviation_
        P[length - 1 , length - 1] = 1 * self.weight_fem_pos_deviation_
        P[length - 2 , length - 1] = -2 * self.weight_fem_pos_deviation_
        P[length - 2 , length - 2] = 5 * self.weight_fem_pos_deviation_

        for i in range(2 , length - 2):
            P[i , i] = 6 * self.weight_fem_pos_deviation_
        for i in range(2 , length - 1):
            P[i - 1, i] = -4 * self.weight_fem_pos_deviation_
        for i in range(2 , length):
            P[i - 2, i] = 1 * self.weight_fem_pos_deviation_

        # with np.printoptions(precision=0):
        #     print(P)

        P = P / self.weight_fem_pos_deviation_
        P = sparse.csc_matrix(P)

        #set q matrix , from calculateOffset
        q = np.zeros(length)

        #set Bound(upper/lower bound) matrix , add constraints for x
        #from CalculateAffineConstraint

        #Config limit with (0.1,0.5) , Here I set a constant 0.2 
        bound = 0.5
        A = np.zeros((length,length))
        for i in range(length):
            A[i, i] = 1
        A = sparse.csc_matrix(A)
        lx = np.array(x_array) - bound
        ux = np.array(x_array) + bound
        ly = np.array(y_array) - bound
        uy = np.array(y_array) + bound

        #solve
        prob = osqp.OSQP()
        prob.setup(P,q,A,lx,ux)
        res = prob.solve()
        opt_x = res.x

        prob.update(l=ly, u=uy)
        res = prob.solve()
        opt_y = res.x

        return opt_x, opt_y

    def calcKappa(self, x_array: List,y_array: List) -> Tuple[List, List]:
        s_array = []
        k_array = []
        if(len(x_array) != len(y_array)):
            return(s_array , k_array)

        length = len(x_array)
        temp_s = 0.0
        s_array.append(temp_s)
        for i in range(1 , length):
            temp_s += np.sqrt(np.square(y_array[i] - y_array[i - 1]) + np.square(x_array[i] - x_array[i - 1]))
            s_array.append(temp_s)

        xds,yds,xdds,ydds = [],[],[],[]
        for i in range(length):
            if i == 0:
                xds.append((x_array[i + 1] - x_array[i]) / (s_array[i + 1] - s_array[i]))
                yds.append((y_array[i + 1] - y_array[i]) / (s_array[i + 1] - s_array[i]))
            elif i == length - 1:
                xds.append((x_array[i] - x_array[i-1]) / (s_array[i] - s_array[i-1]))
                yds.append((y_array[i] - y_array[i-1]) / (s_array[i] - s_array[i-1]))
            else:
                xds.append((x_array[i+1] - x_array[i-1]) / (s_array[i+1] - s_array[i-1]))
                yds.append((y_array[i+1] - y_array[i-1]) / (s_array[i+1] - s_array[i-1]))
        for i in range(length):
            if i == 0:
                xdds.append((xds[i + 1] - xds[i]) / (s_array[i + 1] - s_array[i]))
                ydds.append((yds[i + 1] - yds[i]) / (s_array[i + 1] - s_array[i]))
            elif i == length - 1:
                xdds.append((xds[i] - xds[i-1]) / (s_array[i] - s_array[i-1]))
                ydds.append((yds[i] - yds[i-1]) / (s_array[i] - s_array[i-1]))
            else:
                xdds.append((xds[i+1] - xds[i-1]) / (s_array[i+1] - s_array[i-1]))
                ydds.append((yds[i+1] - yds[i-1]) / (s_array[i+1] - s_array[i-1]))
        for i in range(length):
            k_array.append((xds[i] * ydds[i] - yds[i] * xdds[i]) / (np.sqrt(xds[i] * xds[i] + yds[i] * yds[i]) * (xds[i] * xds[i] + yds[i] * yds[i]) + 1e-6));
        return s_array, k_array


class Trajectory:
    def __init__(self,length,width):
        self.length=length
        self.width=width
        self.t = []
        self.l = []
        self.l_d = []
        self.l_dd = []
        self.l_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []

        self.x = []
        self.y = []
        self.v = []
        self.a = []
        self.yaw = []
        self.k = []
        self.c = []
        self.ref_kappa = []
        self.vx = []
        self.w = []

        self.v_length = 0
        self.v_width = 0
        
        self.ds = []
        
        self.longi_sample = None
        self.lat_sample = None
        self.rs = None
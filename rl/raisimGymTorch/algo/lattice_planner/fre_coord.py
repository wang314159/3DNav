
from typing import Any, List, Tuple
from typing import List
from scipy.spatial import cKDTree
import numpy as np
from math import *
import tools
import math

class FrenetCoord:
    def __init__(self,x: List, y: List, ds: float = 0.1) -> None:
        self.x = x
        self.y = y
        self.ds = ds
        self.spline2D = tools.Spline2D(self.x,self.y)
        self.s, self.rx, self.ry, self.ryaw, self.rk = \
            self._cal_frenet_coord(ds = self.ds)

        self.kd_tree = cKDTree(np.c_[np.array(self.rx).ravel(),np.array(self.ry).ravel()])#k领域树
        # print("--init frenet time:",time.time())
    
    def expand_ref_line(self):
        ref_x_list = []
        ref_y_list = []
        ref_x_list.extend(self.x)
        ref_y_list.extend(self.y)  

        delta_x = self.x[-1] - self.x[-2]
        delta_y = self.y[-1] - self.y[-2]

        for i in range(1,100):
            ref_x_list.append(i * delta_x + self.x[-1])
            ref_y_list.append(i * delta_y + self.y[-1])

        delta_x = self.x[0] - self.x[1]
        delta_y = self.y[0] - self.y[1]

        for i in range(1,20):
            ref_x_list.insert(0, ref_x_list[0] + delta_x)
            ref_y_list.insert(0, ref_y_list[0] + delta_y)

        return ref_x_list,ref_y_list


    def _cal_frenet_coord(self, ds: float = 0.01) -> Tuple[List, List, List, List, List, Any, Any, Any]:
        s = list(np.arange(0, self.spline2D.s[-1], ds))
        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = self.spline2D.calc_position(i_s)#根据s算xy
            rx.append(ix)
            ry.append(iy)
            ryaw.append(self.spline2D.calc_yaw(i_s))#每个点的切线
            rk.append(self.spline2D.calc_curvature(i_s))#每个点的曲率

        return s, rx, ry, ryaw, rk
    
    def heading(self, x:float, y:float)->float:#轨迹点投影到参考线上的切线向量
        s, l = self.xy2sl(x, y)
        x_delta, y_delta = self.sl2xy(s+0.1, 0)
        x, y = self.sl2xy(s, 0)
        return math.atan2(y_delta-y, x_delta-x)
    
    def xy2sl(self, x:float, y:float)->Tuple[float, float]:#输入参考线外的x，y坐标，找到在参考线上的匹配点。算出s和d
        rs = self.find_nearest_rs(x, y)#找最邻近点
        rx, ry, rtheta, rkappa, rdkappa = self.find_nearest_frenet_point(rs)
        s_condition, l_condition = self.cartesian_to_frenet1D(rs, rx, ry, rtheta, x, y)
        return (s_condition[0], l_condition[0])
    
    def sl2xy(self, s:float, l:float)->Tuple[float, float]:
        rx, ry, rtheta, rkappa, rdkappa = self.find_nearest_frenet_point(s)
        x, y = self.frenet_to_cartesian1D(s, rx, ry, rtheta, s, l)
        return (x, y)

    def velocity_xy2sl(self, x:float, y:float, v: float, h: float)->Tuple[float, float]:
        rs = self.find_nearest_rs(x, y)
        rx, ry, rtheta, rkappa, rdkappa = self.find_nearest_frenet_point(rs)
        s_condition, l_condition = self.cartesian_to_frenet2D(rs, rx, ry, rtheta, rkappa, x, y, v, h)
        return (s_condition[1], l_condition[1])
    def velocity_xy2sl_(self, x:float, y:float, v: float, h: float)->Tuple[float, float]:
        rs = self.find_nearest_rs(x, y)
        rx, ry, rtheta, rkappa, rdkappa = self.find_nearest_frenet_point(rs)
        s_condition, l_condition = self.cartesian_to_frenet2D(rs, rx, ry, rtheta, rkappa, x, y, v, h)
        return (s_condition, l_condition)
    def acceleration_xy3sl(self,x:float, y:float, v: float, a: float, h: float,k:float)->Tuple[float, float]:
        rs = self.find_nearest_rs(x, y)
        rx, ry, rtheta, rkappa, rdkappa = self.find_nearest_frenet_point(rs)
        s_condition, l_condition=self.cartesian_to_frenet3D( rs, rx, ry, rtheta, rkappa, rdkappa,\
                    x, y, v, a, h, k)
        return s_condition, l_condition
    
    def velocity_sl2xy(self, s:float, s_d: float, l:float, l_d:float)->Tuple[float, float]:
        rx, ry, rtheta, rkappa, rdkappa = self.find_nearest_frenet_point(s)
        x, y, v, theta = self.frenet_to_cartesian2D(s, rx, ry, rtheta, rkappa, s, s_d, l, l_d)
        return (v*cos(theta), v*sin(theta))
    

    #[x,y]→[s,l]
    def cartesian_to_frenet1D(self, rs: float, rx: float, ry: float, rtheta: float, x: float, y: float) -> Tuple[float, float]:#根据xy算sl
        s_condition = np.zeros(1)
        d_condition = np.zeros(1)
        
        dx = x - rx
        dy = y - ry
        
        cos_theta_r = cos(rtheta)
        sin_theta_r = sin(rtheta)
        
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d_condition[0] = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)    
        
        s_condition[0] = rs
        
        return s_condition, d_condition

    #[s,l]→[x,y]
    def frenet_to_cartesian1D(self, rs: float, rx: float, ry: float, rtheta: float, s: float, l: float) -> Tuple[float, float]:
        if fabs(rs - s)>= 1.0e-1:
            print("The reference point s and s_condition[0] don't match")
        
        cos_theta_r = cos(rtheta)
        sin_theta_r = sin(rtheta)
        
        x = rx - sin_theta_r * l
        y = ry + cos_theta_r * l    

        return x, y

    def cartesian_to_frenet2D(self, rs: float, rx: float, ry: float, rtheta: float, rkappa: float, x: float, y: float, v: float, theta: float) -> Tuple[Any, Any]:#根据xy算s和s点，l和l撇
        s_condition = np.zeros(2)
        d_condition = np.zeros(2)
        
        dx = x - rx
        dy = y - ry
        
        cos_theta_r = cos(rtheta)
        sin_theta_r = sin(rtheta)
        
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d_condition[0] = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)
        
        delta_theta = theta - rtheta
        tan_delta_theta = tan(delta_theta)
        cos_delta_theta = cos(delta_theta)
        
        one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
        d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
        
        
        s_condition[0] = rs
        s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d

        return s_condition, d_condition

    def frenet_to_cartesian2D(self, rs: float, rx: float, ry: float, rtheta: float, rkappa: float, s: float, s_diff: float, l: float, l_diff: float):
        if fabs(rs - s)>= 1.0e-1:
            print("The reference point s and s_condition[0] don't match")
        
        cos_theta_r = cos(rtheta)
        sin_theta_r = sin(rtheta)
        
        x = rx - sin_theta_r * l
        y = ry + cos_theta_r * l

        one_minus_kappa_r_d = 1 - rkappa * l
        tan_delta_theta = l_diff / one_minus_kappa_r_d
        delta_theta = atan2(l_diff, one_minus_kappa_r_d)
        cos_delta_theta = cos(delta_theta)
        
        theta = self.NormalizeAngle(delta_theta + rtheta)    
        
        d_dot = l_diff * s_diff
        
        v = sqrt(one_minus_kappa_r_d * one_minus_kappa_r_d * s_diff * s_diff + d_dot * d_dot)   

        return x, y, v, theta

    def frenet_to_cartesian3D(self, rs: float, rx: float, ry: float, rtheta: float, rkappa: float, rdkappa: float,\
                        s: float, s_d: float, s_dd: float, l: float, l_d: float, l_dd: float) -> Tuple[float, float, float, float, float, float]:
        if fabs(rs - s)>= 1.0e-6:
            print("The reference point s and s_condition[0] don't match")
        
        cos_theta_r = cos(rtheta)
        sin_theta_r = sin(rtheta)
        
        x = rx - sin_theta_r * l
        y = ry + cos_theta_r * l

        one_minus_kappa_r_d = 1 - rkappa * l
        tan_delta_theta = l_d / one_minus_kappa_r_d
        delta_theta = atan2(l_d, one_minus_kappa_r_d)
        cos_delta_theta = cos(delta_theta)
        
        theta = self.NormalizeAngle(delta_theta + rtheta)
        kappa_r_d_prime = rdkappa * l + rkappa * l_d
            
        kappa = ((((l_dd + kappa_r_d_prime * tan_delta_theta) *
                    cos_delta_theta * cos_delta_theta) /
                        (one_minus_kappa_r_d) +
                    rkappa) *
                cos_delta_theta / (one_minus_kappa_r_d))
        
        
        d_dot = l_d * l_d
        
        v = sqrt(one_minus_kappa_r_d * one_minus_kappa_r_d * s_d * s_d + d_dot * d_dot)
        
        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * (kappa) - rkappa     
        a = (s_dd * one_minus_kappa_r_d / cos_delta_theta +
            s_d * s_d / cos_delta_theta *
                (l_d * delta_theta_prime - kappa_r_d_prime))
        return x, y, v, a, theta, kappa

    def cartesian_to_frenet3D(self, rs: float, rx: float, ry: float, rtheta: float, rkappa: float, rdkappa: float,\
                    x: float, y: float, v: float, a: float, theta: float, kappa: float) -> Tuple[Any, Any]:#从xy求出s、ds、dds，l、l_prime、l_prime_prime
        s_condition = np.zeros(3)
        d_condition = np.zeros(3)
        
        dx = x - rx
        dy = y - ry
        
        cos_theta_r = cos(rtheta)
        sin_theta_r = sin(rtheta)
        
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d_condition[0] = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)
        
        delta_theta = theta - rtheta
        tan_delta_theta = tan(delta_theta)
        cos_delta_theta = cos(delta_theta)
        
        one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
        d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
        
        kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
        
        d_condition[2] = (-kappa_r_d_prime * tan_delta_theta + 
        one_minus_kappa_r_d / cos_delta_theta / cos_delta_theta *
            (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))
        
        s_condition[0] = rs
        s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d
        
        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
        s_condition[2] = ((a * cos_delta_theta -
                        s_condition[1] * s_condition[1] *
                        (d_condition[1] * delta_theta_prime - kappa_r_d_prime)) /
                            one_minus_kappa_r_d)
        return s_condition, d_condition
    
    def find_nearest_rs(self, x: float, y: float) -> float:
        dis,index = self.kd_tree.query([x,y],k = 1)
        rs = self.s[index]
        return rs

    def find_nearest_frenet_point(self, rs: float):
        rs = np.clip(rs,0,self.s[-1])#超出上下限取最大或最小
        rx = self.spline2D.sx.calc(rs)
        ry = self.spline2D.sy.calc(rs)
        dx = self.spline2D.sx.calcd(rs)
        dy = self.spline2D.sy.calcd(rs)
        ddx = self.spline2D.sx.calcdd(rs)
        ddy = self.spline2D.sy.calcdd(rs)
        dddx = self.spline2D.sx.calcddd(rs)
        dddy = self.spline2D.sy.calcddd(rs)
        rtheta = atan2(dy, dx)
        rkappa = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))
        rdkappa = dx * dddy - dy * dddx

        return rx, ry, rtheta, rkappa, rdkappa
    def find_nearest_frenet_point_2(self, rs: float):
        rs = np.clip(rs,0,self.s[-1])#超出上下限取最大或最小
        rx = self.spline2D.sx.calc(rs)
        ry = self.spline2D.sy.calc(rs)
        dx = self.spline2D.sx.calcd(rs)
        dy = self.spline2D.sy.calcd(rs)
        v=math.sqrt(dx**2+dy**2)
        ddx = self.spline2D.sx.calcdd(rs)
        ddy = self.spline2D.sy.calcdd(rs)
        a=math.sqrt(ddx**2+ddy**2)
        dddx = self.spline2D.sx.calcddd(rs)
        dddy = self.spline2D.sy.calcddd(rs)
        rtheta = atan2(dy, dx)
        rkappa = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))
        rdkappa = dx * dddy - dy * dddx

        return rx, ry, v,a,rtheta, rkappa, rdkappa
    def NormalizeAngle(self, angle: float)->float:
        a = fmod(angle+np.pi, 2*np.pi)
        if a < 0.0:
            a += (2.0*np.pi)        
        return a - np.pi
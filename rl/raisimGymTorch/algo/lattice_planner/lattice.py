

# from numba import jit
import math
from math import *

import numpy as np
import sys
sys.path.append("/home/ws/raisim/raisimProject/3DNav/rl/raisimGymTorch/algo/lattice_planner")
import tools
import fre_coord
# from macro_definition import *
import time
import matplotlib.pyplot as plt
# from env.IDM import IDM_al


def generate_cruise_longitudinal_speed(c_s, c_sd, c_sdd, g_sd, g_sdd,
                                       t) -> tools.QuarticPolynomial:
    """纵向巡航
    function:
        generate lattice longitudinal speed for crusie Scenerio
    input:
        start state info:
            c_s: current longitudinal s 当前纵向偏移
            c_sd: current longitudinal speed 当前纵向偏移速度
            c_sdd: current longitudinal acc 当前纵向偏移加速度
        end state info:
            g_sd: goal longitudinal speed 目标纵向偏移速度
            g_sdd: goal longitudinal acc 目标纵向偏移加速度
        t: time to goal longitudinal speed  到达目标速度的时间
    output:
        longitudinal speed
    """
    longi_speed = tools.QuarticPolynomial(c_s, c_sd, c_sdd, g_sd, g_sdd, t)
    return longi_speed
def generate_cruise_longitudinal_speed_troch(c_s, c_sd, c_sdd, g_sd, g_sdd,
                                       t) -> tools.QuarticPolynomial_torch:
    """纵向巡航
    function:
        generate lattice longitudinal speed for crusie Scenerio
    input:
        start state info:
            c_s: current longitudinal s 当前纵向偏移
            c_sd: current longitudinal speed 当前纵向偏移速度
            c_sdd: current longitudinal acc 当前纵向偏移加速度
        end state info:
            g_sd: goal longitudinal speed 目标纵向偏移速度
            g_sdd: goal longitudinal acc 目标纵向偏移加速度
        t: time to goal longitudinal speed  到达目标速度的时间
    output:
        longitudinal speed
    """
    longi_speed = tools.QuarticPolynomial_torch(c_s, c_sd, c_sdd, g_sd, g_sdd, t)
    return longi_speed
def generate_lateral_path(c_d, c_dd , c_ddd, g_d, g_dd, g_ddd, s) -> tools.QuinticPolynomial:
        """横向
        function:
            generate lattice lateral path
        input:
            start state info:
                c_d: current lateral offset 当前横向偏移
                c_dd: current lateral speed 当前横向偏移速度
                c_ddd: current lateral acc 当前横向偏移加速度
            end state info:
                g_d: goal lateral offset 目标横向偏移
                g_dd: goal lateral speed 目标横向偏移速度
                g_ddd: goal lateral acc 目标横向偏移加速度
            s: Longitudinal distance to goal lateral offset 到达横向偏移量的纵向距离
        output:
            lateral path 输出：横向的轨迹点
        """
        lat_path = tools.QuinticPolynomial(c_d, c_dd, c_ddd, g_d, g_dd, g_ddd, s)
        return lat_path


def generate_lateral_path(c_d, c_dd , c_ddd, g_dd, g_ddd, s) -> tools.QuinticPolynomial:
        """横向
        function:
            generate lattice lateral path
        input:
            start state info:
                c_d: current lateral offset 当前横向偏移
                c_dd: current lateral speed 当前横向偏移速度
                c_ddd: current lateral acc 当前横向偏移加速度
            end state info:
                g_d: goal lateral offset 目标横向偏移
                g_dd: goal lateral speed 目标横向偏移速度
                g_ddd: goal lateral acc 目标横向偏移加速度
            s: Longitudinal distance to goal lateral offset 到达横向偏移量的纵向距离
        output:
            lateral path 输出：横向的轨迹点
        """
        lat_path = tools.QuarticPolynomial(c_d, c_dd, c_ddd, g_dd, g_ddd, s)
        return lat_path


def generate_lateral_path_torch(c_d, c_dd, c_ddd, g_d, g_dd, g_ddd, s) -> tools.QuinticPolynomial_torch:
    """横向
    function:
        generate lattice lateral path
    input:
        start state info:
            c_d: current lateral offset 当前横向偏移
            c_dd: current lateral speed 当前横向偏移速度
            c_ddd: current lateral acc 当前横向偏移加速度
        end state info:
            g_d: goal lateral offset 目标横向偏移
            g_dd: goal lateral speed 目标横向偏移速度
            g_ddd: goal lateral acc 目标横向偏移加速度
        s: Longitudinal distance to goal lateral offset 到达横向偏移量的纵向距离
    output:
        lateral path 输出：横向的轨迹点
    """
    lat_path = tools.QuinticPolynomial_torch(c_d, c_dd, c_ddd, g_d, g_dd, g_ddd, s)
    return lat_path

class obstacle:
    def __init__(self,):
        self.start_s=0
        self.end_s=0
        self.start_l=0
        self.end_l=0

class PathTimeMap:
    def __init__(self):

        self.obstacle=[]






def isCollision_(dx,dy,shift_x,shift_y,T_yaw,half_length_,half_width_,obs_half_length,obs_half_width):
        cos_heading_ = math.cos(T_yaw)
        sin_heading_ = math.sin(T_yaw)
        obs_psi = math.atan(dy / (dx+0.001))
        obs_cos_heading = math.cos(obs_psi)
        obs_sin_heading = math.sin(obs_psi)
        dx1 = cos_heading_ * half_length_
        dy1 = sin_heading_ * half_length_
        dx2 = sin_heading_ * half_width_
        dy2 = -cos_heading_ * half_width_
        dx3 = obs_cos_heading * obs_half_length
        dy3 = obs_sin_heading * obs_half_length
        dx4 = obs_sin_heading * obs_half_width
        dy4 = -obs_cos_heading * obs_half_width
        a = (abs(shift_x * cos_heading_ + shift_y * sin_heading_) <=
             abs(dx3 * cos_heading_ + dy3 * sin_heading_) + abs(
            dx4 * cos_heading_ + dy4 * sin_heading_) + half_length_)
        b = abs(shift_x * sin_heading_ - shift_y * cos_heading_) <= abs(dx3 * sin_heading_ - dy3 * cos_heading_) + abs(
            dx4 * sin_heading_ - dy4 * cos_heading_) + half_width_
        c = abs(shift_x * obs_cos_heading + shift_y * obs_sin_heading) <= abs(
            dx1 * obs_cos_heading + dy1 * obs_sin_heading) + abs(
            dx2 * obs_cos_heading + dy2 * obs_sin_heading) + obs_half_length
        d = abs(shift_x * obs_sin_heading - shift_y * obs_cos_heading) <= abs(
            dx1 * obs_sin_heading - dy1 * obs_cos_heading) + abs(
            dx2 * obs_sin_heading - dy2 * obs_cos_heading) + obs_half_width
        if a & b & c & d:  # 如果True就代表碰撞了
            return True
        else:
            return False



class LatticeTrajectoryGenerator:
    def __init__(self,env_num,t,dt:float) -> None:
        self.t=t
        self.dt=dt
        self.env_num=env_num
        self.ref_line=fre_coord.FrenetCoord([0,1],[0,0],0.01)
        self.LatticeTrajectoryHighSpeedPostProcess = LatticeTrajectoryHighSpeedPostProcess(self.t,self.ref_line,self.dt)

    def update(self,v,a) -> None:
        self.a=np.clip(a,-2,2)
        self.Trajectory=[]
        self.v=v
        self.generate_trajectory()

    def generate_trajectory(self):
        for i in range(self.env_num):
            longi_speed = self.generate_cruise_longitudinal_speed( 0, self.v[i], self.a[i][0], self.v[i]+self.a[i][0]*self.t, self.a[i][0], self.t)
            lat_path = self.generate_lateral_path(0, 0, self.a[i][1], self.a[i][1]*self.t, self.a[i][1], 
                                                longi_speed.calc_point(self.t))
            self.Trajectory.append(self.LatticeTrajectoryHighSpeedPostProcess.output_trajectory(longi_speed=longi_speed, lat_path=lat_path,
                                                                                        rs=0,length=0,width=0)) 


    def generate_lateral_path(self, c_d, c_dd, c_ddd, g_dd, g_ddd, s) -> tools.QuinticPolynomial:
        """横向
        function:
            generate lattice lateral path
        input:
            start state info:
                c_d: current lateral offset 当前横向偏移
                c_dd: current lateral speed 当前横向偏移速度
                c_ddd: current lateral acc 当前横向偏移加速度
            end state info:
                g_d: goal lateral offset 目标横向偏移
                g_dd: goal lateral speed 目标横向偏移速度
                g_ddd: goal lateral acc 目标横向偏移加速度
            s: Longitudinal distance to goal lateral offset 到达横向偏移量的纵向距离
        output:
            lateral path 输出：横向的轨迹点
        """
        lat_path = tools.QuarticPolynomial(c_d, c_dd, c_ddd, g_dd, g_ddd, s)
        return lat_path
    
    
    def generate_cruise_longitudinal_speed(self, c_s: float, c_sd: float, c_sdd: float, g_sd: float, g_sdd: float, t: float) -> tools.QuarticPolynomial:
        """纵向巡航
        function:
            generate lattice longitudinal speed for crusie Scenerio
        input:
            start state info:
                c_s: current longitudinal s 当前纵向偏移
                c_sd: current longitudinal speed 当前纵向偏移速度
                c_sdd: current longitudinal acc 当前纵向偏移加速度
            end state info:
                g_sd: goal longitudinal speed 目标纵向偏移速度
                g_sdd: goal longitudinal acc 目标纵向偏移加速度
            t: time to goal longitudinal speed  到达目标速度的时间
        output:
            longitudinal speed
        """
        longi_speed = tools.QuarticPolynomial(c_s, c_sd, c_sdd, g_sd, g_sdd, t)
        return longi_speed


class LatticeTrajectoryHighSpeedPostProcess:
    def __init__(self, t_trj: float, fre_coord: fre_coord.FrenetCoord, dt: float = 0.04) -> None:
        """
        function:
            Constructor
        Input:
            t_trj:trajectory duration
            dt: Time interval between two track points
            fre_coord: frenet coordinate
        Output:
            None
        """
        self.t_trj = t_trj
        self.fre_coord = fre_coord
        self.dt = dt
    
    def update(self, t_trj: float, fre_coord: fre_coord.FrenetCoord, dt: float = 0.04) -> None:
        """
        function:
            Update the parameters of the class
        Input:
            t_trj:trajectory duration
            dt: Time interval between two track points
            fre_coord: frenet coordinate
        Output:
            None
        """
        self.t_trj = t_trj
        self.fre_coord = fre_coord
        self.dt = dt

    def output_trajectory(self, longi_speed, lat_path, rs: float,length,width, sample_t:float=None) -> tools.Trajectory:
        """
        function:
            Combine lateral path and longitudinal speed to output full trajectory
        input:
            longi_speed: longitudinal speed 纵向速度
            lat_path: lateral_path 横向轨迹轨迹
        output:
            trajectory 轨迹
        """
        if sample_t is None:
            sample_t = self.t_trj
        else:
            # sample_t = ((int)(sample_t * 10))/10
            sample_t = round(sample_t,2)
        trajectory = tools.Trajectory(length,width)
        num = round(sample_t/self.dt)
        trajectory.yaw.append(0)
        # print("num:",num)
        for t in np.linspace(0.0, sample_t, num, endpoint = False):
            s=longi_speed.calc_point(t)
            trajectory.t.append(t)#时间
            trajectory.s.append(s)#纵向位置
            trajectory.s_d.append(longi_speed.calc_first_derivative(t))#纵向速度
            trajectory.s_dd.append(longi_speed.calc_second_derivative(t))#加速度
            trajectory.s_ddd.append(longi_speed.calc_third_derivative(t))#加加速度

            trajectory.l.append(lat_path.calc_point(s-rs))#横向位置
            trajectory.l_d.append(lat_path.calc_first_derivative(s-rs))#横向速度
            trajectory.l_dd.append(lat_path.calc_second_derivative(s-rs))#加速度
            trajectory.l_ddd.append(lat_path.calc_third_derivative(s-rs))#加加速度
            
            rx, ry, rtheta, rkappa, rdakappa = self.fre_coord.find_nearest_frenet_point(trajectory.s[-1])
            x, y, v, a, theta, kappa = self.fre_coord.frenet_to_cartesian3D(
                trajectory.s[-1],rx,ry,rtheta,rkappa,rdakappa,\
                    trajectory.s[-1],trajectory.s_d[-1],trajectory.s_dd[-1],\
                        trajectory.l[-1],trajectory.l_d[-1],trajectory.l_dd[-1]
            )
            
            trajectory.x.append(x)
            trajectory.y.append(y)
            trajectory.v.append(v)
            trajectory.a.append(a)
            trajectory.k.append(kappa)
        
            if len(trajectory.x) > 1:
                dx = trajectory.x[-1] - trajectory.x[-2]
                dy = trajectory.y[-1] - trajectory.y[-2]
                if dx != 0 and dy != 0:
                    trajectory.yaw.append(atan2(dy, dx))#斜率的度数
                    trajectory.ds.append(math.sqrt(dx ** 2 + dy ** 2))#两个差分时刻的距离
                else:
                    trajectory.yaw.append( self.fre_coord.heading(trajectory.x[-1], trajectory.y[-1]) )
                    # trajectory.yaw.append(last_yaw)
                    trajectory.ds.append(0)
            if len(trajectory.yaw) > 1:
                trajectory.c.append(
                    (abs(trajectory.yaw[-1]) - abs(trajectory.yaw[-2])) / (trajectory.ds[-1] + 1e-9)
                )#参考线曲率
        for i in range(len(trajectory.yaw)-1):
            trajectory.w.append((trajectory.yaw[i+1]-trajectory.yaw[i])/self.dt)
        trajectory.w.append(0)
        trajectory.longi_sample = longi_speed.sample_param
        trajectory.lat_sample = lat_path.sample_param
        # trajectory.rs = rs + trajectory.s[1]
        return trajectory
    
#test
if __name__ == '__main__':
    gen = LatticeTrajectoryGenerator(1,0.1,0.005)
    gen.update([0.1],np.array([[0.2,-0.1]]))
    gen.generate_trajectory()
    # print("y\t:",gen.Trajectory.y)
    # print("x\t:",gen.Trajectory.x)
    print("v\t:",gen.Trajectory[0].v)
    print("w\t:",gen.Trajectory[0].w)
    print("yaw\t:",gen.Trajectory[0].yaw)
    # gen.Trajectory.yaw.append(gen.Trajectory.yaw[-1])
    # gen.Trajectory.w.append(gen.Trajectory.w[-1])
    print(len(gen.Trajectory[0].v))
    print(len(gen.Trajectory[0].w))
    # plt.plot(gen.Trajectory.x,label="x")
    # plt.plot(gen.Trajectory.y,label="y")
    plt.plot(gen.Trajectory[0].x,gen.Trajectory[0].y)
    # plt.plot(gen.Trajectory.x)
    # plt.plot(gen.Trajectory.w,label="w")
    plt.show()
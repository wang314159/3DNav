

# from numba import jit
import math
from math import *

import numpy as np
from lattice_planner import tools
from lattice_planner import fre_coord
# from macro_definition import *
import time
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
    def __init__(self,tracks,static_track,dt:float,init_frame) -> None:
        self.tracks=tracks#车辆的原始轨迹信息
        self.dt=dt
        self.init_frame=init_frame#车辆开始被接管的那一帧
        self.static_track=static_track#被接管车辆的静态信息
        self.OrinObsTracks=[]
        self.ContObsTracks_lattice=[]
        self.ContObsTracks_clip=[]
        self.ref_line=fre_coord.FrenetCoord((tracks[BBOX][init_frame-self.static_track[INITIAL_FRAME]:,0]+tracks[BBOX][0,2]*0.5).tolist(),
                                            (tracks[BBOX][init_frame-self.static_track[INITIAL_FRAME]:,1]+tracks[BBOX][0,3]*0.5).tolist(),0.01)
        # if init_frame-self.static_track[INITIAL_FRAME]+100>=len(tracks[BBOX]):
        #      self.targetpoint,_ = self.ref_line.xy2sl(tracks[BBOX][-1, 0],
        #                                             tracks[BBOX][-1, 1])
        # else:
        #      self.targetpoint,_ =self.ref_line.xy2sl(tracks[BBOX][init_frame-self.static_track[INITIAL_FRAME]+100,0],
        #                                          tracks[BBOX][init_frame-self.static_track[INITIAL_FRAME]+100,1])
        self.psi=np.zeros(len(self.tracks[Y_VELOCITY]))
        for i in range(len(self.tracks[Y_VELOCITY])):
            self.psi[i]=math.atan(self.tracks[Y_VELOCITY][i]/abs(self.tracks[X_VELOCITY][i]))
        self.LatticeTrajectoryHighSpeedPostProcess = LatticeTrajectoryHighSpeedPostProcess(2.2,self.ref_line,0.04)
        self.Trajectory=[]
        self.current_frame=init_frame-self.static_track[INITIAL_FRAME]
        self.plantime=4.0
        self.initV=sqrt(self.tracks[X_VELOCITY][self.current_frame]**2+self.tracks[Y_VELOCITY][self.current_frame]**2)
        self.initA=sqrt(self.tracks[X_ACCELERATION][self.current_frame]**2+self.tracks[Y_ACCELERATION][self.current_frame]**2)
        self.init_theta=acos(self.tracks[X_VELOCITY][self.current_frame]/ self.initV)

        rkappa = (self.tracks[X_ACCELERATION][self.current_frame] * self.tracks[X_VELOCITY][self.current_frame] -
                  self.tracks[X_ACCELERATION][self.current_frame] * self.tracks[Y_VELOCITY][self.current_frame]) / \
                 ((self.tracks[X_VELOCITY][self.current_frame] ** 2 + self.tracks[Y_VELOCITY][self.current_frame] ** 2) ** (3 / 2))

        s,l=self.ref_line.acceleration_xy3sl(self.tracks[BBOX][self.current_frame][0]+self.tracks[BBOX][self.current_frame][2]*0.5,self.tracks[BBOX][self.current_frame][1]+self.tracks[BBOX][self.current_frame][3]*0.5,self.initV,self.initA,self.init_theta,rkappa)
        self.init_state_s=dict({"state":np.array([s[0],s[1],s[2]]),"time":0})
        self.init_state_l=dict({"state":np.array([l[0],l[1],l[2]]),"path":0})
        self.lon_trajectory_bundle=[]
        self.lat_trajectory_bundle=[]
        self.out_Trajectory=None
        self.lateralSample()



    def update_frame(self,init_frame,trajectory):
        self.init_frame=init_frame
        if init_frame>self.tracks[FRAME][-1]:
            self.current_frame = self.tracks[FRAME][-1] - self.static_track[INITIAL_FRAME]
        else:
            self.current_frame = init_frame - self.static_track[INITIAL_FRAME]
        self.initV = trajectory.v[5]
        self.initA = trajectory.a[5]
        self.init_theta =trajectory.yaw[4]
        self.init_state_s = dict({"state": np.array([trajectory.s[5], trajectory.s_d[5], trajectory.s_dd[5]]), "time": 0})
        self.init_state_l = dict({"state": np.array([trajectory.l[5], trajectory.l_d[5], trajectory.l_dd[5]]), "path": 0})
        self.lon_trajectory_bundle = []
        self.lat_trajectory_bundle = []
        self.lateralSample()
        self.Trajectory = []





    def cruiseSample(self):
        end_s_conditions=[]
        ref_speed=sqrt(self.tracks[X_VELOCITY][self.current_frame]**2+self.tracks[X_VELOCITY][self.current_frame]**2)
        theta=acos(self.tracks[X_VELOCITY][self.current_frame]/ref_speed)
        vs,pl=self.ref_line.velocity_xy2sl(self.tracks[BBOX][self.current_frame][0],self.tracks[BBOX][self.current_frame][1],ref_speed,theta)
        for i in range(5):
            for j in range(10):
                if i==0:
                    end_s_conditions.append(dict({"state":np.array([0,(vs+5)*((i+1)/10),0]),"time":0.01}))
                else:
                    end_s_conditions.append(dict({"state": np.array([0, (vs+5)*((i+1)/10) + j, 0]), "time": 2 * (i/ 5)}))

        for end_s_condition in end_s_conditions:
            longi_cruise_speed=self.generate_cruise_longitudinal_speed(self.init_state_s["state"][0],
                                                                       self.init_state_s["state"][1],
                                                                       self.init_state_s["state"][2],
                                                                       end_s_condition["state"][1],
                                                                       end_s_condition["state"][2],
                                                                       end_s_condition["time"])
            self.lon_trajectory_bundle.append(longi_cruise_speed)


    def followSample(self,current_frame):
        obstacless=tools.Obstacles(self.OrinObsTracks,self.ContObsTracks_lattice,self.ContObsTracks_clip,current_frame)
        end_s_conditions=obstacless.get_end_conditions(self.ref_line)
        for end_s_condition in end_s_conditions:
            if end_s_condition["state"][0]<self.init_state_s["state"][0]:
                continue
            if end_s_condition["state"][0] > self.init_state_s["state"][0]+20:
                continue
            log_follow_path=self.generate_follow_longitudinal_speed(self.init_state_s["state"][0],
                                                             self.init_state_s["state"][1],
                                                             self.init_state_s["state"][2],
                                                             end_s_condition["state"][0],
                                                             end_s_condition["state"][1],
                                                             end_s_condition["state"][2],
                                                             end_s_condition["time"])
            self.lon_trajectory_bundle.append(log_follow_path)

    def lateralSample(self):
        end_d_conditions=[]
        end_d_candidates = [0.0]
        end_s_candidates = [80.0]
        for  s in end_s_candidates :
            for  d in end_d_candidates:
                end_d_state = np.array([d,0,0])
                end_d_conditions.append(dict({"state":end_d_state,"path":s}))
        for end_d_condition in end_d_conditions:
            lat_path=self.generate_lateral_path(self.init_state_l["state"][0],
                                                self.init_state_l["state"][1],
                                                self.init_state_l["state"][2],
                                                end_d_condition["state"][0],
                                                end_d_condition["state"][1],
                                                end_d_condition["state"][2],
                                                end_d_condition["path"])
            self.lat_trajectory_bundle.append(lat_path)

    def LatticeTrajectory2dCombin(self):

        for long in self.lon_trajectory_bundle:
            if long.T<0.2:
                continue
            for lat in self.lat_trajectory_bundle:
                Trajector=self.LatticeTrajectoryHighSpeedPostProcess.output_trajectory(longi_speed=long,lat_path=lat,rs=self.init_state_s["state"][0],length=self.tracks[BBOX][0,2],width=self.tracks[BBOX][0,3],sample_t=long.T)
                #longi_speed, lat_path
                # if min(Trajector.x) > 0 and max(Trajector.x) < 410 and min(Trajector.y) > 0 and abs(max(
                #         Trajector.y)-min(Trajector.y)) < 5:
                cost = self.CalcCost(long, lat)
                self.Trajectory.append(dict({"combined_trajectory": Trajector, "cost": cost}))

        # 根据cost从小到大排序
        self.Trajectory = sorted(self.Trajectory, key=lambda x: x['cost'])
        #self.Trajectory=reversed(self.Trajectory)

    def IDMtraject(self):

        #s,l=self.ref_line.xy2sl(self.ContObsTracks_clip[0][0]["x"],self.ContObsTracks_clip[0][0]["y"])
        s,l=self.ref_line.velocity_xy2sl_(self.ContObsTracks_clip[0][0]["x"],self.ContObsTracks_clip[0][0]["y"],self.ContObsTracks_clip[0][0]["v"],self.ContObsTracks_clip[0][0]["psi"])
        if self.current_frame+5>=len(self.tracks[FRAME]):
            v = (self.tracks[X_VELOCITY][-1] ** 2 + self.tracks[Y_VELOCITY][
                -1] ** 2) ** 0.5
            h = math.acos(self.tracks[X_VELOCITY][-1] / v)
            sd, ld = self.ref_line.velocity_xy2sl_(self.tracks[BBOX][-1, 0],
                                                   self.tracks[BBOX][-1, 1], v, h)
        else:
            v=(self.tracks[X_VELOCITY][self.current_frame+5]**2+self.tracks[Y_VELOCITY][self.current_frame]**2)**0.5
            h=math.acos(self.tracks[X_VELOCITY][self.current_frame+5]/v)
            sd,ld=self.ref_line.velocity_xy2sl_(self.tracks[BBOX][self.current_frame+5,0],self.tracks[BBOX][self.current_frame+5,1],v,h)
        if s[0]-self.ContObsTracks_clip[0][0]["len"]>self.init_state_s["state"][0]:
            a=IDM_al(s[1],s[0]-self.ContObsTracks_clip[0][0]["len"]-self.init_state_s["state"][0],self.init_state_s["state"][1],sd[1])
        else:
            a = IDM_al(self.init_state_s["state"][1], 100,
                       self.init_state_s["state"][1], sd[1])
        a=np.clip(a,-6,5)
        end_s_condition=np.array([0,self.init_state_s["state"][1]+a,0])
        long=self.generate_cruise_longitudinal_speed(self.init_state_s["state"][0],
                                                                       self.init_state_s["state"][1],
                                                                       self.init_state_s["state"][2],
                                                                       end_s_condition[1],
                                                                       end_s_condition[2],
                                                                       0.4)
        lat=self.lat_trajectory_bundle[0]
        Trajector = self.LatticeTrajectoryHighSpeedPostProcess.output_trajectory(longi_speed=long, lat_path=lat,
                                                                                 rs=self.init_state_s["state"][0],
                                                                                 length=self.tracks[BBOX][0, 2],
                                                                                 width=self.tracks[BBOX][0, 3],
                                                                                 sample_t=long.T)
        return Trajector



    def LatticeTrajectory2dCombin_2(self):

        for long in self.lon_trajectory_bundle:
            if long.T<0.2:
                continue
            for lat in self.lat_trajectory_bundle:
                Trajector=self.LatticeTrajectoryHighSpeedPostProcess.output_trajectory(longi_speed=long,lat_path=lat,rs=self.init_state_s["state"][0],length=self.tracks[BBOX][0,2],width=self.tracks[BBOX][0,3],sample_t=long.T)
                    #longi_speed, lat_path
                    # if min(Trajector.x) > 0 and max(Trajector.x) < 410 and min(Trajector.y) > 0 and abs(max(
                    #         Trajector.y)-min(Trajector.y)) < 5:
                cost = self.CalcCost(long, lat)
                self.Trajectory.append(dict({"combined_trajectory": Trajector, "cost": cost}))

        # 根据cost从小到大排序
        self.Trajectory = sorted(self.Trajectory, key=lambda x: x['cost'])
    def ValidTrajectory(self,Trajectory):

        for i in range(len(Trajectory.a)):
            if Trajectory.a[i]<-10 or Trajectory.a[i]>6:
                return "A_laji",i
            if Trajectory.v[i]<0 or Trajectory.v[i]>50:
                return "V_laji",i
            if Trajectory.k[i]<-0.1979 or Trajectory.k[i]>0.1979:
                return "K_laji",i
            if i>21:
                return "VALID",None
            #if i<len(Trajectory.a)-1:
                # lon_jerk = (Trajectory.a[i+1]- Trajectory.a[i])/ (Trajectory.t[i+1]- Trajectory.t[i])
                # if lon_jerk<-4 or lon_jerk>2:
                #     return "laji"
                # lat_a = Trajectory.v[i+1] * Trajectory.v[i+1] *Trajectory.k[i+1]
                # if lat_a<-4 or lat_a>4:
                #     return "laji"
        return "VALID",None
        print (" ")

    def InCollision(self,Trajectory):
        half_length_=self.tracks[BBOX][0,2]*0.5
        half_width_=self.tracks[BBOX][0,3]*0.5
        for obs in self.OrinObsTracks:
            frame=obs[FRAME][0]
            obs_half_length=obs[BBOX][0,2]*0.5
            obs_half_width=obs[BBOX][0,3]*0.5
            for i in range(len(Trajectory.t)-2):
                if i>self.init_frame-frame+i-1:
                    break
                shift_x = obs[BBOX][self.init_frame-frame+i,0] - Trajectory.x[i+1]
                shift_y = obs[BBOX][self.init_frame-frame+i,1] - Trajectory.y[i+1]
                dx=obs[BBOX][self.init_frame-frame+i+1,0]-obs[BBOX][self.init_frame-frame+i,0]
                dy=obs[BBOX][self.init_frame-frame+i+1,1]-obs[BBOX][self.init_frame-frame+i,1]
                a=self.isCollision(dx,dy,shift_x,shift_y,Trajectory.yaw[i],half_length_,half_width_,obs_half_length,obs_half_width)
                if a:
                    return True,"OrinObsTracks,{}".format(i)
                #dx dy shift_x shift_y  yaw
        for obs in self.ContObsTracks_lattice:
            if obs is None or len(obs.x)<7:
                continue
            obs_half_length=obs.length*0.5
            obs_half_width=obs.width*0.5
            for i in range(len(Trajectory.t)-2):
                if i+6>len(obs.x)-1 or i>15:
                    break
                shift_x=obs.x[5+i]-Trajectory.x[i+1]
                shift_y=obs.y[5+i]-Trajectory.y[i+1]
                dx=obs.x[6+i]-obs.x[5+i]
                dy=obs.x[6+i]-obs.x[5+i]
                a= self.isCollision(dx, dy, shift_x, shift_y, Trajectory.yaw[i], half_length_, half_width_,
                                        obs_half_length, obs_half_width)
                if a:
                    return True,"ContObsTracks_lattice,{}".format(i)
        for obs in self.ContObsTracks_clip:
            obs_half_length = obs[0]["len"] * 0.5
            obs_half_width = obs[0]["wid"] * 0.5
            for i in range(4):
                shift_x=obs[i]["x"]-Trajectory.x[i+1]
                shift_y=obs[i]["y"]-Trajectory.y[i+1]
                dx=obs[i+1]["x"]-obs[i]["x"]
                dy=obs[i+1]["y"]-obs[i]["y"]
                a= self.isCollision(dx, dy, shift_x, shift_y, Trajectory.yaw[i], half_length_, half_width_,
                                        obs_half_length, obs_half_width)
                if a:
                    return True,"ContObsTracks_clip,{}".format(i)

        return False,None

    def isCollision(self,dx,dy,shift_x,shift_y,T_yaw,half_length_,half_width_,obs_half_length,obs_half_width):
        cos_heading_ = math.cos(T_yaw)
        sin_heading_ = math.sin(T_yaw)
        obs_psi = math.atan(dy /( dx+0.001))
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
        a = abs(shift_x * cos_heading_ + shift_y * sin_heading_) <= abs(dx3 * cos_heading_ + dy3 * sin_heading_) + abs(
            dx4 * cos_heading_ + dy4 * sin_heading_) + half_length_
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



    def TrajectoryChoose(self):
        result_list =[]
        i=0
        for Trajectory in self.Trajectory:
            result ,j= self.ValidTrajectory(Trajectory["combined_trajectory"])
            i=i+1
            if (result != "VALID"):
                result_list.append({"inde":i,"result":result,"frame":j})
                continue
            result,state=self.InCollision(Trajectory["combined_trajectory"])
            if (result):
                result_list.append({"inde": i, "result":"InCollision", "statej": state})
                continue
            self.out_Trajectory=Trajectory["combined_trajectory"]
            return Trajectory["combined_trajectory"]
        print(" ")





    def CalcCost(self,lon_trajectory,lat_trajectory):
        lon_objective_cost =self.LonObjectiveCost(lon_trajectory)#和参考速度比较计算cost
        lon_jerk_cost = self.LonComfortCost(lon_trajectory)#舒适度
        lon_collision_cost = self.LonCollisionCost(lon_trajectory)#纵向碰撞，距离越近cost越大
        centripetal_acc_cost = self.CentripetalAccelerationCost(lon_trajectory)
        lat_offset_cost = self.LatOffsetCost(lon_trajectory, lat_trajectory)
        lat_comfort_cost = self.LatComfortCost(lon_trajectory, lat_trajectory)
        return lon_objective_cost * 5 + lon_jerk_cost * 1 + lon_collision_cost * 10 +  centripetal_acc_cost * 1.5 + lat_offset_cost * 1.0 +  lat_comfort_cost * 5



    def LonObjectiveCost(self,lon_trajectory):
            t_max=lon_trajectory.T
            dist_s=lon_trajectory.calc_point(t_max)-lon_trajectory.calc_point(0)
            speed_cost_sqr_sum = 0.0
            speed_cost_weight_sum = 0.0
            for  i in range(int(t_max/0.04)):
                t = (i) * 0.04
                if self.current_frame+i>len(self.tracks[FRAME])-1:
                    sv,lv=self.ref_line.velocity_xy2sl(self.tracks[BBOX][len(self.tracks[FRAME])-1,0],self.tracks[BBOX][len(self.tracks[FRAME])-1,1],self.tracks[X_VELOCITY][len(self.tracks[FRAME])-1],self.psi[len(self.tracks[FRAME])-1])
                else:
                    sv, lv = self.ref_line.velocity_xy2sl(self.tracks[BBOX][self.current_frame + i, 0],
                                                          self.tracks[BBOX][self.current_frame + i, 1],
                                                          self.tracks[X_VELOCITY][self.current_frame + i],
                                                          self.psi[self.current_frame + i])

                cost =  sv- lon_trajectory.calc_first_derivative(t)
                speed_cost_sqr_sum += t * t *abs(cost)
                speed_cost_weight_sum += t * t
            speed_cost =speed_cost_sqr_sum / (speed_cost_weight_sum + 0.001)
            dist_travelled_cost = 1.0 / (1.0 + dist_s)
            return (speed_cost * 1 +dist_travelled_cost * 10)/11

    def LonComfortCost(self,lon_trajectory):
        cost_sqr_sum = 0.0
        cost_abs_sum = 0.0
        t=0
        while t<lon_trajectory.T:
            t += 0.04
            t = round(t, 2)
            jerk = lon_trajectory.calc_third_derivative(t)
            cost = jerk/2
            cost_sqr_sum += cost * cost
            cost_abs_sum += abs(cost)
        return cost_sqr_sum / (cost_abs_sum + 0.001)
    def obsfitting(self):
        self.followCarBottemPointSpline_s = []
        # self.followCarBottemPointSpline_l = []
        # self.followCarBottemPointSpline_psi = []
        t_list = np.linspace(0.0, 2.04, 51, endpoint=False).tolist()
        for obs in self.OrinObsTracks:
            s_list = []
            l_list=[]
            psi_list=[]
            frame = obs[FRAME][0]
            for i in range(51):
                if frame - self.init_frame + i > len(obs[BBOX]):
                    break
                s, l = self.ref_line.xy2sl(obs[BBOX][frame - self.init_frame + i, 0],
                                           obs[BBOX][frame - self.init_frame + i, 1])
                s_list.append(s)
                if i== len(obs[BBOX]):
                    psi_list.append(0)
                else:
                    dx=obs[BBOX][frame - self.init_frame + i+1, 0]-obs[BBOX][frame - self.init_frame + i, 0]
                    dy = obs[BBOX][frame - self.init_frame + i + 1, 1] - obs[BBOX][frame - self.init_frame + i, 1]
                    psi_list.append(math.atan(dy/dx))
            self.followCarBottemPointSpline_s.append(tools.Spline(t_list[0:i], s_list))
            # self.followCarBottemPointSpline_l.append(tools.Spline(t_list[0:i], l_list))
            # self.followCarBottemPointSpline_psi.append(tools.Spline(t_list[0:i], psi_list))
        for obs in self.ContObsTracks_lattice:
            if obs is None:
                continue
            s_list = []
            l_list = []
            psi_list = []
            for i in range(51):
                if i > len(obs.x)-1:
                    break
                s, l = self.ref_line.xy2sl(obs.x[i], obs.y[i])
                s_list.append(s)
                if i==len(obs.x)-1:
                    psi_list.append(psi_list[-1])
                else:
                    dx=obs.x[i+1]-obs.x[i]
                    dy=obs.y[i+1]-obs.y[i]
                    psi_list.append(math.atan(dy / dx))
            #self.followCarBottemPointSpline_psi.append(tools.Spline(t_list[0:i], psi_list))
            self.followCarBottemPointSpline_s.append(tools.Spline(t_list[0:i], s_list))
            #self.followCarBottemPointSpline_l.append(tools.Spline(t_list[0:i], l_list))
        s_list = []
        l_list = []
        psi_list = []
        for tracks in self.ContObsTracks_clip:
            for track in tracks:
                s, l = self.ref_line.xy2sl(track["x"], track["y"])
                l_list.append(l)
                s_list.append(s)
                psi_list.append(track["psi"])
        self.followCarBottemPointSpline_s.append(tools.Spline(t_list[1:6], s_list))
        # self.followCarBottemPointSpline_l.append(tools.Spline(t_list[0:4], l_list))
        # self.followCarBottemPointSpline_psi.append(tools.Spline(t_list[0:4], psi_list))

    def  LonCollisionCost(self,lon_trajectory):
        cost_sqr_sum = 0.0
        cost_abs_sum = 0.0
        for followCarBottemPointSpline in self.followCarBottemPointSpline_s:
            t=0
            while t<lon_trajectory.T:
                if t>followCarBottemPointSpline.x[-1]:
                    break
                t += 0.04
                t = round(t, 2)
                traj_s = lon_trajectory.calc_point(t)
                sigma = 0.5
                dist = 0.0
                if t>followCarBottemPointSpline.x[-1]:
                    break
                if (traj_s < followCarBottemPointSpline.calc(t) - 5):
                     dist =  followCarBottemPointSpline.calc(t) - 5 - traj_s
                elif (traj_s > followCarBottemPointSpline.calc(t) + 5):
                    dist = traj_s - followCarBottemPointSpline.calc(t) - 5
                cost = math.exp(-dist * dist / (2.0 * sigma * sigma))
                cost_sqr_sum += cost * cost
                cost_abs_sum += cost
        return cost_sqr_sum / (cost_abs_sum + 0.001)
    def CentripetalAccelerationCost(self,lon_trajectory):

        centripetal_acc_sum = 0.0
        centripetal_acc_sqr_sum = 0.0
        t=0
        while t<lon_trajectory.T:
            t += 0.04
            t = round(t, 2)
            s = lon_trajectory.calc_point( t)
            v = lon_trajectory.calc_first_derivative(t)
            rx, ry, rtheta, rkappa, rdkappa =self.ref_line.find_nearest_frenet_point(s)
            centripetal_acc = v * v * rkappa
            centripetal_acc_sum += abs(centripetal_acc)
            centripetal_acc_sqr_sum += centripetal_acc * centripetal_acc
        return centripetal_acc_sqr_sum /(centripetal_acc_sum + 0.001)

    def LatOffsetCost(self,lon_trajectory,lat_trajectory):
        lat_offset_start = lat_trajectory.calc_point(0)
        cost_sqr_sum = 0.0
        cost_abs_sum = 0.0
        s_value=lon_trajectory.calc_point(lon_trajectory.T)
        for s in range(int(s_value)):
            lat_offset = lat_trajectory.calc_point(s)
            cost = lat_offset/3
            if (lat_offset * lat_offset_start < 0.0):
                cost_sqr_sum += cost * cost * 10
                cost_abs_sum +=(cost) * 10
            else:
                cost_sqr_sum += cost * cost * 1
                cost_abs_sum += abs(cost) * 1
        return cost_sqr_sum / (cost_abs_sum + 0.001)

    def LatComfortCost(self,lon_trajectory, lat_trajectory):
        max_cost = 0.0
        t=0
        while t<lon_trajectory.T:
            t += 0.04
            t = round(t, 2)
            s = lon_trajectory.calc_point(t)
            s_dot = lon_trajectory.calc_first_derivative(t)
            s_dotdot = lon_trajectory.calc_second_derivative(t)
            relative_s = s - self.init_state_s["state"][0]
            l_prime = lat_trajectory.calc_first_derivative(relative_s)
            l_primeprime = lat_trajectory.calc_second_derivative(relative_s)
            cost = l_primeprime * s_dot * s_dot + l_prime * s_dotdot
            max_cost = max(max_cost, abs(cost))
        return max_cost


    def generate_lateral_path(self, c_d: float, c_dd: float, c_ddd: float, g_d: float, g_dd: float, g_ddd: float, s:float) -> tools.QuinticPolynomial:
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
    
       
    def generate_follow_longitudinal_speed(self, c_s: float, c_sd: float, c_sdd: float, g_s: float ,g_sd: float, g_sdd: float, t: float) -> tools.QuinticPolynomial:
        """纵向跟车
        function:
            generate lattice longitudinal speed for follow Scenerio
        input:
            start state info:
                c_s: current longitudinal s
                c_sd: current longitudinal speed
                c_sdd: current longitudinal acc
            end state info:
                g_s: goal longitudinal s
                g_sd: goal longitudinal speed
                g_sdd: goal longitudinal acc
            t: time to goal longitudinal speed/s
        output:
            longitudinal speed
        """
        longi_speed = tools.QuinticPolynomial(c_s, c_sd, c_sdd, g_s, g_sd, g_sdd, t)
        return longi_speed
    
    
    def generate_stop_longitudinal_speed(self, c_s: float, c_sd: float, c_sdd: float, g_s: float ,g_sd: float, g_sdd: float, t: float) -> tools.QuinticPolynomial:
        """纵向停车
        function:
            generate lattice longitudinal speed for follow Scenerio
        input:
            start state info:
                c_s: current longitudinal s
                c_sd: current longitudinal speed
                c_sdd: current longitudinal acc
            end state info:
                g_s: goal longitudinal s
                g_sd: goal longitudinal speed
                g_sdd: goal longitudinal acc
            t: time to goal longitudinal s 到达目标位置的时间
        output:
            longitudinal speed
        """
        longi_speed = tools.QuinticPolynomial(c_s, c_sd, c_sdd, g_s, g_sd, g_sdd, t)
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

    def output_trajectory_2(self, longi_speed, rs: float, length, width,
                          sample_t: float = None) -> tools.Trajectory:
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
            sample_t = round(sample_t + self.dt, 2)
        trajectory = tools.Trajectory(length, width)
        num = round(sample_t / self.dt)
        for t in np.linspace(0.0, sample_t, num, endpoint=False):
            s = longi_speed.calc_point(t)
            trajectory.t.append(t)  # 时间
            trajectory.s.append(s)  # 纵向位置
            trajectory.s_d.append(longi_speed.calc_first_derivative(t))  # 纵向速度
            trajectory.s_dd.append(longi_speed.calc_second_derivative(t))  # 加速度
            trajectory.s_ddd.append(longi_speed.calc_third_derivative(t))  # 加加速度
            rx, ry, v,a,rtheta, rkappa, rdakappa=self.fre_coord.find_nearest_frenet_point_2(s)
            s,l=self.fre_coord.acceleration_xy3sl(rx,ry,v,a,rtheta,rkappa)
            trajectory.l.append(l[0])  # 横向位置
            trajectory.l_d.append(l[1])  # 横向速度
            trajectory.l_dd.append(l[2])  # 加速度
            #trajectory.l_ddd.append(lat_path.calc_third_derivative(s - rs))  # 加加速度

            rx, ry, rtheta, rkappa, rdakappa = self.fre_coord.find_nearest_frenet_point(trajectory.s[-1])
            x, y, v, a, theta, kappa = self.fre_coord.frenet_to_cartesian3D(
                trajectory.s[-1], rx, ry, rtheta, rkappa, rdakappa, \
                trajectory.s[-1], trajectory.s_d[-1], trajectory.s_dd[-1], \
                trajectory.l[-1], trajectory.l_d[-1], trajectory.l_dd[-1]
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
                    trajectory.yaw.append(atan2(dy, dx))  # 斜率的度数
                    trajectory.ds.append(math.sqrt(dx ** 2 + dy ** 2))  # 两个差分时刻的距离
                else:
                    trajectory.yaw.append(self.fre_coord.heading(trajectory.x[-1], trajectory.y[-1]))
                    # trajectory.yaw.append(last_yaw)
                    trajectory.ds.append(0)
            if len(trajectory.yaw) > 1:
                trajectory.c.append(
                    (abs(trajectory.yaw[-1]) - abs(trajectory.yaw[-2])) / (trajectory.ds[-1] + 1e-9)
                )  # 参考线曲率

        trajectory.longi_sample = longi_speed.sample_param
        #trajectory.lat_sample = lat_path.sample_param
        # trajectory.rs = rs + trajectory.s[1]
        return trajectory
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
            sample_t = round(sample_t+self.dt,2)
        trajectory = tools.Trajectory(length,width)
        num = round(sample_t/self.dt)
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
                
        trajectory.longi_sample = longi_speed.sample_param
        trajectory.lat_sample = lat_path.sample_param
        # trajectory.rs = rs + trajectory.s[1]
        return trajectory
    

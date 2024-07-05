//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <algorithm>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "parameters.hpp"
#include <pcl/common/common_headers.h>
#include "lidar.hpp"
#include "elevationMap.hpp"

namespace raisim {


class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1){
    
    // std::cout<<"init env"<<std::endl;
    params_.update(cfg);

    std::uniform_real_distribution<double> unif_(0,params_.goalthresh[0]*2);
    for(int i=0;i<2;i++){
      double temp=0;
      while(abs(temp)<params_.goalthresh[1])
        temp = unif_(gen_) - params_.goalthresh[0];
      goalpos[i]=temp;
    }
    goalpos[0] = abs(goalpos[0]);
    // std::cout<<"goalpos"<<goalpos[0]<<" "<<goalpos[1]<<std::endl;
    // goalpos[0]=4;//params_.goalpos[0];
    // goalpos[1]=4;//params_.goalpos[1];
    /// create world
    world_ = std::make_unique<raisim::World>();
    mapheight = params_.map_param[0],mapwidth = params_.map_param[1];
    raisim::TerrainProperties terrainProperties;
    terrainProperties.frequency = 0.2;
    terrainProperties.zScale = 2.0;
    terrainProperties.xSize = mapheight;
    terrainProperties.ySize = mapwidth;
    terrainProperties.xSamples = 100;
    terrainProperties.ySamples = 100;
    terrainProperties.fractalOctaves = 3;
    terrainProperties.fractalLacunarity = 2.0;
    terrainProperties.fractalGain = 0.25;

  // auto hm = world.addHeightMap("../data/height3.png", 0, 0, 3, 3, 0.000001, 0);
    hm_ = world_->addHeightMap(0, 0, terrainProperties);
    // hm_ = world_->addHeightMap(resourceDir_ + params_.map_path, 0, 0, mapheight, 
    // mapwidth, params_.map_param[2], params_.map_param[3]);
    hm_->setAppearance("soil2");

    /// add objects
    nanocar_ = world_->addArticulatedSystem(resourceDir_+ params_.robot_urdf);
    nanocar_->setName("nanocar");
    nanocar_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();
    lidar_.init(params_.scanSize[0], params_.scanSize[1], visualizable_);
    /// get robot data
    gcDim_ = nanocar_->getGeneralizedCoordinateDim();
    gvDim_ = nanocar_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);vTarget4_.setZero(nJoints_); pTarget4_.setZero(nJoints_);
    damping.setZero(gvDim_);
    damping.tail(nJoints_).setConstant(1.);
    nanocar_->setJointDamping(damping);
    /// this is nominal configuration of nanocar
    // gc_init_ << -20, -20, 0.2, 1, 0, 0, 0;
    for (int i = 0; i < 7; i++) gc_init_[i]  = params_.gc_init[i];
    // gc_init_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(params_.gc_init, 7);
    // std::cout<<"gc_init"<<gc_init_[0]<<"gc_"<<gc_[0]<<std::endl;
    nanocar_->setGeneralizedCoordinate(gc_init_);
    nanocar_->setGeneralizedVelocity(gv_init_);
    
    init_dist=sqrt((gc_init_[0]-goalpos[0])*(gc_init_[0]-goalpos[0])+(gc_init_[1]-goalpos[1])*(gc_init_[1]-goalpos[1]));
    last_dist=dist=init_dist;
    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(30.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    nanocar_->setPdGains(jointPgain, jointDgain);
    nanocar_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 11 ;//+ lidar_.e_.GetHeightVec().size();
    actionDim_ = 2; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    bodyLinearVel_.setZero(),bodyAngularVel_.setZero();
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    wheelIndices_.insert(nanocar_->getBodyIdx("/front_left_wheel_link"));
    wheelIndices_.insert(nanocar_->getBodyIdx("/front_right_wheel_link"));
    wheelIndices_.insert(nanocar_->getBodyIdx("/back_left_wheel_link"));
    wheelIndices_.insert(nanocar_->getBodyIdx("/back_right_wheel_link"));

    server_ = nullptr;
    /// visualize if it is the first environment
    if (visualizable_) {
      // std::cout<<"visualize"<<std::endl;
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      goal_ = server_->addVisualCylinder("goal",0.1,2);
      goal_->setPosition({goalpos[0],goalpos[1],1});
      // scans_ = server_->addInstancedVisuals("scan points",
      //                                     raisim::Shape::Box,
      //                                     {0.01, 0.01, 0.01},
      //                                     {1,0,0,1},
      //                                     {0,1,0,1});
      // scans_->resize(params_.scanSize[0]*params_.scanSize[1]);
      server_->launchServer();
      server_->focusOn(nanocar_);
    }
    // std::cout<<"init env ok"<<std::endl;
  }

  void init() final { }

  void reset() final {
    // std::cout<<"reset"<<std::endl;
    nanocar_->setState(gc_init_, gv_init_);
    pTarget_.setZero(); vTarget_.setZero();vTarget4_.setZero(); pTarget4_.setZero();
    last_dist=init_dist;
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    // std::cout<<action[0]<<" "<<action[1]<<std::endl;
    // std::cout<<"step"<<std::endl;
    v=action[0],w=action[1];
    // v=0.5,w=0;
    v=std::clamp(v,-1.5,1.5),w=std::clamp(w,-1.0,1.0);
    w=std::clamp(w,-abs(v/3),abs(v/3));
    
    vr = (v + w*d)/r;
    vl = (v - w*d)/r;
    theta = atan(l*w/(v+0.0001));
    // if(visualizable_){
    //   std::cout<<"v"<<v<<"w"<<w<<std::endl;
    //   std::cout<<"theta"<<theta<<"vr"<<vr<<"vl"<<vl<<std::endl;
    // }
    vTarget4_ <<0,vl, 0,vr, vl, vr;
    pTarget4_ = gc_.tail(nJoints_);
    pTarget4_ += vTarget4_*(control_dt_ + simulation_dt_);
    pTarget4_[0] = theta, pTarget4_[2] = theta;
    vTarget_.tail(nJoints_) = vTarget4_;
    pTarget_.tail(nJoints_) = pTarget4_;
    nanocar_->setPdTarget(pTarget_,vTarget_);
    // std::cout<<"begin integrate"<<std::endl;
    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      // std::cout<<"integrate"<<std::endl;
      world_->integrate();
      lidar_.scan(world_, server_, nanocar_);
      if(server_) server_->unlockVisualizationServerMutex();
    }
    
    // if(visualizable_) lidar_.visualize(scans_);
    // std::cout<<"observe"<<std::endl;
    updateObservation();
    // std::cout<<"observation"<<std::endl;

    // rewards_.record("torque", nanocar_->getGeneralizedForce().squaredNorm());
    // rewards_.record("forwardVel", std::min(1.5,bodyLinearVel_[0]));
    // rewards_.record("AngularVel", -abs(bodyAngularVel_[2])); 
    rewards_.record("distance", -dist/init_dist);
    rewards_.record("orientation", M_PI/3-abs(delta_theta));
    // if(visualizable_)
    // std::cout<<bodyLinearVel_[0]<<" "<<dist<<" "<<last_dist<<std::endl;
    if(dist<0.2){
      std::cout<<"reach goal"<<std::endl;
      rewards_.record("reach", init_dist*init_dist);
    }else{
      rewards_.record("reach", 0);
    }
    last_dist=dist;
    return rewards_.sum();
  }

  void updateObservation() {
    nanocar_->getState(gc_, gv_);
    // std::cout<<"observation"<<std::endl;
    dist=sqrt((gc_[0]-goalpos[0])*(gc_[0]-goalpos[0])+(gc_[1]-goalpos[1])*(gc_[1]-goalpos[1]));
    quat_db[0] = gc_[3]; quat_db[1] = gc_[4]; quat_db[2] = gc_[5]; quat_db[3] = gc_[6];
    raisim::quatToEulerVec(quat_db, euler);
    while(euler[2]>M_PI) euler[2]-=2*M_PI;
    while(euler[2]<-M_PI) euler[2]+=2*M_PI;
    goal_theta = atan2(goalpos[1]-gc_[1],goalpos[0]-gc_[0]);
    delta_theta = goal_theta-euler[2];
    if(delta_theta>M_PI) delta_theta-=2*M_PI;
    if(delta_theta<-M_PI) delta_theta+=2*M_PI;
    // if(visualizable_) std::cout<<euler[2]<<" "<<goal_theta<<" "<<delta_theta<<std::endl;
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << dist,delta_theta,//relative position to goal
	      bodyLinearVel_[0],//linearvelocity
        bodyAngularVel_[2],//angularvelocity
        euler[0],euler[1],euler[2];//orientation
        //lidar_.e_.GetHeightVec(); 
    // std::cout<<"observation end"<<std::endl;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: nanocar_->getContacts())
      if(wheelIndices_.find(contact.getlocalBodyIndex()) == wheelIndices_.end())
        return true;
    // std::cout<<gc_[0]<<" "<<std::endl;
    if(abs(gc_[0])>(mapheight/2) || abs(gc_[1])>(mapwidth/2)){
      return true;
    }
    if(dist<0.2){
      terminalReward = 100;
      return true;
    }
	    
    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* nanocar_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget4_, vTarget_,vTarget4_,damping;
  double terminalRewardCoeff_ = -1000;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  double quat_db[4],euler[3],goal_theta,delta_theta;
  Eigen::Vector3d bodyLinearVel_,bodyAngularVel_;
  std::set<size_t> wheelIndices_;
  Parameters params_;
  HeightMap* hm_;
  double goalpos[2]={0,0};
  double mapheight,mapwidth;
  double vr,vl,r=0.0335,d=0.08725,v,w,R,l=0.14353,theta,dist,last_dist=dist,init_dist=dist;
  // raisim::InstancedVisuals* scans_;
  raisim::Visuals* goal_;
  lidar lidar_;
  

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}


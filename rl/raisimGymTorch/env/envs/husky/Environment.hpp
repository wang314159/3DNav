//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
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
    goalpos[0]=params_.goalpos[0];
    goalpos[1]=params_.goalpos[1];
    /// create world
    world_ = std::make_unique<raisim::World>();
    mapheight = params_.map_param[0],mapwidth = params_.map_param[1];
    raisim::TerrainProperties terrainProperties;
    terrainProperties.frequency = 0.2;
    terrainProperties.zScale = 3.0;
    terrainProperties.xSize = mapheight;
    terrainProperties.ySize = mapwidth;
    terrainProperties.xSamples = 50;
    terrainProperties.ySamples = 50;
    terrainProperties.fractalOctaves = 3;
    terrainProperties.fractalLacunarity = 2.0;
    terrainProperties.fractalGain = 0.25;
    hm_ = world_->addHeightMap(0, 0, terrainProperties);
    // hm_ = world_->addHeightMap(resourceDir_ + params_.map_path, 0, 0, mapheight, 
    // mapwidth, params_.map_param[2], params_.map_param[3]);
    hm_->setAppearance("soil2");

    /// add objects
    husky_ = world_->addArticulatedSystem(resourceDir_+ params_.robot_urdf);
    husky_->setName("husky");
    husky_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();
    //lidar_.init(params_.scanSize[0], params_.scanSize[1], visualizable_);
    /// get robot data
    gcDim_ = husky_->getGeneralizedCoordinateDim();
    gvDim_ = husky_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);vTarget4_.setZero(gvDim_); pTarget4_.setZero(nJoints_);

    /// this is nominal configuration of husky
    // gc_init_ << -20, -20, 0.2, 1, 0, 0, 0;
    gc_init_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(params_.gc_init, 7);
    husky_->setGeneralizedCoordinate(gc_init_);
    husky_->setGeneralizedVelocity(gv_init_);
    

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    husky_->setPdGains(jointPgain, jointDgain);
    husky_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 11 ;//+ lidar_.e_.GetHeightVec().size();
    actionDim_ = 2; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    bodyLinearVel_.setZero();
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    wheelIndices_.insert(husky_->getBodyIdx("front_left_wheel_link"));
    wheelIndices_.insert(husky_->getBodyIdx("front_right_wheel_link"));
    wheelIndices_.insert(husky_->getBodyIdx("rear_left_wheel_link"));
    wheelIndices_.insert(husky_->getBodyIdx("rear_right_wheel_link"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      // scans_ = server_->addInstancedVisuals("scan points",
      //                                     raisim::Shape::Box,
      //                                     {0.01, 0.01, 0.01},
      //                                     {1,0,0,1},
      //                                     {0,1,0,1});
      // scans_->resize(params_.scanSize[0]*params_.scanSize[1]);
      server_->launchServer();
      server_->focusOn(husky_);
    }
    // std::cout<<"init env ok"<<std::endl;
  }

  void init() final { }

  void reset() final {
    husky_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    // std::cout<<"step"<<std::endl;
    /// action scaling
    // pTarget12_ = action.cast<double>();
    // pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    // pTarget12_ += actionMean_;
    // pTarget_.tail(nJoints_) = pTarget12_;

    // husky_->setPdTarget(pTarget_, vTarget_);
    // std::cout<<"step begin"<<std::endl;
    // std::cout<<action[0]<<" "<<action[1]<<std::endl;
    // std::cout<<"???????????"<<std::endl;
    vr = action[0] + action[1]*R/r;
    vl = action[0] - action[1]*R/r;
    vTarget4_ << vl, vr, vl, vr;
    // std::cout<<"step begin2"<<std::endl;
    husky_->getState(gc_,gv_);
    pTarget4_=gc_.tail(nJoints_);
    pTarget4_ += vTarget4_*(control_dt_+simulation_dt_);
    vTarget_.tail(nJoints_) = vTarget4_;
    pTarget_.tail(nJoints_) = pTarget4_;
    husky_->setPdTarget(pTarget_,vTarget_);
    // husky_->setGeneralizedForce({0, 0, 0, 0, 0, 0, action[0], action[1], action[2], action[3]});
    // std::cout<<"begin integrate"<<std::endl;
    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-5); i++){
      if(server_) server_->lockVisualizationServerMutex();
      // std::cout<<"integrate"<<std::endl;
      world_->integrate();
      // lidar_.scan(world_, server_, husky_);
      if(server_) server_->unlockVisualizationServerMutex();
    }
    
    // std::cout<<lidar_.e_.GetHeightVec()[77]<<std::endl;
    // if(visualizable_) lidar_.visualize(scans_);
    // std::cout<<"observe"<<std::endl;
    updateObservation();
    // std::cout<<"observation"<<std::endl;

    double dist=sqrt((gc_[0]-goalpos[0])*(gc_[0]-goalpos[0])+(gc_[1]-goalpos[1])*(gc_[1]-goalpos[1]));
    rewards_.record("torque", husky_->getGeneralizedForce().squaredNorm());
    rewards_.record("forwardVel", std::min(4.0,bodyLinearVel_[0]));
    rewards_.record("distance", -dist/(double)(1.5*params_.map_param[0]));
    // rewards_.record("zmove", -bodyLinearVel_[2]);
    //rewards_.record("roll", -abs(obDouble_[1]));
    //rewards_.record("pitch", -abs(obDouble_[2]));
    // std::cout<<"step c++ end"<<std::endl;
    return rewards_.sum();
  }

  void updateObservation() {
    husky_->getState(gc_, gv_);
    // Eigen::Quaterniond quaternion(gc_[3], gc_[4], gc_[5], gc_[6]);
    // Euler = quaternion.toRotationMatrix().eulerAngles(2, 1, 0);
    // std::cout<<Init_Euler<<std::endl<<std::endl;
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[0] - goalpos[0],gc_[1] - goalpos[1],//relative position to goal
	      bodyLinearVel_[0],//linearvelocity
        bodyAngularVel_[2],
        rot.e().row(2).transpose(),
        gv_.tail(4);
        //lidar_.e_.GetHeightVec(); 
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: husky_->getContacts())
      if(wheelIndices_.find(contact.getlocalBodyIndex()) == wheelIndices_.end())
        return true;
    if(abs(gc_[0])>(mapheight/2) || abs(gc_[1])>(mapwidth/2))
	    return true;
    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* husky_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget4_, vTarget_,vTarget4_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d Euler, Init_Euler;
  Eigen::Vector3d bodyLinearVel_,bodyAngularVel_;
  std::set<size_t> wheelIndices_;
  Parameters params_;
  HeightMap* hm_;
  double goalpos[2]={0,0};
  double mapheight,mapwidth;
  double r=0.17775,R=0.2854,vl,vr;
  // raisim::InstancedVisuals* scans_;
  // lidar lidar_;
  

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}


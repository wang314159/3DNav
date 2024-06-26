// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include<pcl/io/ply_io.h>
#include <pcl/common/common_headers.h>
#include "lidar.hpp"
#include <cmath>

#define PI 3.1415926


int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  raisim::TerrainProperties terrainProperties;
  terrainProperties.frequency = 0.2;
  terrainProperties.zScale = 3.0;
  terrainProperties.xSize = 20.0;
  terrainProperties.ySize = 20.0;
  terrainProperties.xSamples = 50;
  terrainProperties.ySamples = 50;
  terrainProperties.fractalOctaves = 3;
  terrainProperties.fractalLacunarity = 2.0;
  terrainProperties.fractalGain = 0.25;

  // auto hm = world.addHeightMap("../data/height3.png", 0, 0, 3, 3, 0.000001, 0);
  auto hm = world.addHeightMap(0, 0, terrainProperties);
  // auto hm = world.addHeightMap("height2.png", 0, 0, 50, 50, 0.00005, 1.5);
  // auto robot = world.addArticulatedSystem(binaryPath.getDirectory() + "../data/husky/husky.urdf");
  auto robot = world.addArticulatedSystem(binaryPath.getDirectory() + "../data/nanocarpro/urdf/nanocarpro.urdf");

  robot->setName("husky");
  hm->setAppearance("soil2");
  int gvDim_ = robot->getDOF();
  int gcDim_ = robot->getGeneralizedCoordinateDim();
  int nJoints = gvDim_ - 6;
  Eigen::VectorXd gc(robot->getGeneralizedCoordinateDim()), gv(robot->getDOF()), damping(robot->getDOF());
  gc.setZero(); gv.setZero();
  gc.segment<7>(0) << -8, -8, 2.2, 1, 0, 0, 0;
  robot->setGeneralizedCoordinate(gc);
  robot->setGeneralizedVelocity(gv);
  damping.setConstant(0);
  damping.tail(nJoints).setConstant(1.);
  robot->setJointDamping(damping);
  world.setGravity({0, 0, -9.81});
  robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  Eigen::VectorXd vtarget(robot->getDOF()),vtarget4(nJoints),ptarget(gcDim_),ptarget4(nJoints);
  ptarget.setZero(), ptarget4.setZero();
  vtarget.setZero();
  Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
  jointPgain.setZero(); jointPgain.tail(nJoints).setConstant(50.0);
  jointPgain[1] = 0, jointPgain[3] = 0;
  jointDgain.setZero(); jointDgain.tail(nJoints).setConstant(0.2);
  jointDgain[1] = 0.001, jointDgain[3] = 0.001;
  robot->setPdGains(jointPgain, jointDgain);
  robot->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
  /// launch raisim server
  raisim::RaisimServer server(&world);

  /// this method should be called before server launch
  // raisim::InstancedVisuals* scans = server.addInstancedVisuals("scan points",
  //                                         raisim::Shape::Box,
  //                                         {0.01, 0.01, 0.01},
  //                                         {1,0,0,1},
  //                                         {0,1,0,1});
  int scanSize1 = 50;
  int scanSize2 = 50;
  lidar lidar_;
  lidar_.init(scanSize1, scanSize2, true, server);
  // scans->resize(scanSize1*scanSize2);
  raisim::Visuals* goal;
  double goalpos[2]={-4,-4},theta_goal=0,theta_delta=0;
  goal=server.addVisualCylinder("goal",0.1,1);
  goal->setPosition({goalpos[0],goalpos[1],1});
  server.launchServer();
  server.focusOn(robot);
  double vr,vl,r=0.0335,d=0.1745/2,v,w,R,l=0.14353,theta;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Vector3d Euler, Init_Euler, bodyLinearVel_, bodyAngularVel_ ;
  Eigen::Vector3d direction;
  Eigen::Quaterniond quaternion_init(1, 0, 0, 0);
  
  Init_Euler = quaternion_init.toRotationMatrix().eulerAngles(2, 1, 0);


  for(int time=0; time<1000000; time++) {
    // std::cout<<"time: "<<time<<std::endl;
    RS_TIMED_LOOP(int(world.getTimeStep()*1e6))
    server.integrateWorldThreadSafe();
    lidar_.scan(world, server,robot);
    // raisim::Vec<3> lidarPos; raisim::Mat<3,3> lidarOri;
    // robot->getFramePosition("imu_joint", lidarPos);
    // robot->getFrameOrientation("imu_joint", lidarOri);

    // for(int i=0; i<scanSize1; i++) {
    //   for (int j = 0; j < scanSize2; j++) {
    //     const double yaw = j * M_PI / scanSize2 * 2 -  M_PI;
    //     double pitch = -(i * 0.4/scanSize1) + 0.25;
    //     const double normInv = 1. / sqrt(pitch * pitch + 1);
    //     direction = {cos(yaw) * normInv, sin(yaw) * normInv, -pitch * normInv};
    //     Eigen::Vector3d rayDirection;
    //     rayDirection = lidarOri.e() * direction;
    //     auto &col = world.rayTest(lidarPos.e(), rayDirection, 30);
    //     if (col.size() > 0) {
          
    //       // std::cout<<col[0].getPosition()<<std::endl;
    //       auto relative_pos = col[0].getPosition() - lidarPos.e();
    //       float length = relative_pos.norm();
    //       if(length>1){
    //         scans->setPosition(i * scanSize2 + j, col[0].getPosition());
    //         if(time==100){
    //           pcl::PointXYZ point ={relative_pos[0],relative_pos[1],relative_pos[2]+0.5};
    //           cloud->push_back(point);
    //         }
    //         scans->setColorWeight(i * scanSize2 + j, std::min(length/15.f, 1.0f));
    //       }
    //       else
    //       scans->setPosition(i*scanSize2+j, {0, 0, 100});
    //     }
    //     else
    //       scans->setPosition(i*scanSize2+j, {0, 0, 100});
    //   }
    // }
    // if(time==100){
    //   pcl::io::savePLYFile("map.ply", *cloud);
    // }
    // robot->setGeneralizedVelocity({0, 0, 0, 0, 0, 0, 20, 20, -20, -20});
    // robot->setGeneralizedVelocity({1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    v=1,w=0.1;
    vr = (v + w*d)/r;
    vl = (v - w*d)/r;
    theta = atan2(l*w,v);
    vtarget4 << 0,vl, 0,vr, vl, vr;
    // vtarget4 << 0,0, 0,0, 1, 1;
    ptarget4 += vtarget4*world.getTimeStep();
    // std::cout<<"theta"<<theta<<"vr"<<vr<<"vl"<<vl<<std::endl;
    ptarget4[0] = theta, ptarget4[2] = theta;
    // std::cout<<ptarget.size()<<std::endl;
    ptarget.tail(nJoints) = ptarget4;
    vtarget.tail(nJoints) = vtarget4;
    robot->setPdTarget(ptarget,vtarget);
    
    // gc = robot->getGeneralizedCoordinate().e();
    robot->getState(gc, gv);
    // Eigen::Quaterniond quaternion(gc[3], gc[4], gc[5], gc[6]);
    // Euler = quaternion.toRotationMatrix().eulerAngles(2, 1, 0);
    raisim::Vec<4> quat;
    double quat_db[4],euler[3];
    quat_db[0] = gc[3]; quat_db[1] = gc[4]; quat_db[2] = gc[5]; quat_db[3] = gc[6];
    raisim::Mat<3,3> rot;
    quat[0] = gc[3]; quat[1] = gc[4]; quat[2] = gc[5]; quat[3] = gc[6];
    raisim::quatToRotMat(quat, rot);
    raisim::quatToEulerVec(quat_db, euler);
    theta_goal = atan2(goalpos[1]-gc[1],goalpos[0]-gc[0]);
    std::cout<<euler[2]<<" "<<theta_goal<<" "<<theta_goal-euler[2]<<std::endl;
    // std::cout<<euler[0]<<" "<<euler[1]<<" "<<euler[2]<<std::endl;
    bodyLinearVel_ = rot.e().transpose() * gv.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv.segment(3, 3);
    // std::cout<<"gc:"<<gc[3]<<" "<<gc[4]<<" "<<gc[5]<<" "<<gc[6]<<std::endl;
    // std::cout<<Euler[0]<<" "<<Euler[1]<<std::endl<<std::endl;
    // std::cout<<rot.e().row(2)<<std::endl;
    // std::cout<<bodyLinearVel_[0]<<" "<<bodyAngularVel_[2]<<std::endl;
    // std::cout<<gc[0]<<std::endl;
    if(fabs(gc[0])>35. || fabs(gc[1])>35.) {
      gc.segment<7>(0) << 0, 0, 2, 1, 0, 0, 0;
      gv.setRandom();
      robot->setState(gc, gv);
    }
  }

  server.killServer();
}

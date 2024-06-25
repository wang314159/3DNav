// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include<pcl/io/ply_io.h>
#include <pcl/common/common_headers.h>
#include "lidar.hpp"


int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  raisim::TerrainProperties terrainProperties;
  terrainProperties.frequency = 0.2;
  terrainProperties.zScale = 2.0;
  terrainProperties.xSize = 50.0;
  terrainProperties.ySize = 50.0;
  terrainProperties.xSamples = 50;
  terrainProperties.ySamples = 50;
  terrainProperties.fractalOctaves = 3;
  terrainProperties.fractalLacunarity = 2.0;
  terrainProperties.fractalGain = 0.25;

  auto hm = world.addHeightMap("../data/height3.png", 0, 0, 10, 10, 0.00001, 0);
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
  gc.segment<7>(0) << -4, -4, 0.2, 1, 0, 0, 0;
  robot->setGeneralizedCoordinate(gc);
  robot->setGeneralizedVelocity(gv);
  damping.setConstant(0);
  damping.tail(nJoints).setConstant(1.);
  robot->setJointDamping(damping);
  // world.setGravity({0, 0, -9.81});
  robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  Eigen::VectorXd vtarget(robot->getDOF()),vtarget4(nJoints),ptarget(gcDim_),ptarget4(nJoints);
  ptarget.setZero(), ptarget4.setZero();
  vtarget.setZero();
  Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
  jointPgain.setZero(); jointPgain.tail(nJoints).setConstant(50.0);
  jointPgain[1] = 0, jointPgain[3] = 0;
  jointDgain.setZero(); jointDgain.tail(nJoints).setConstant(0.2);
  jointDgain[1] = 0.001, jointDgain[3] = 0;
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
  server.launchServer();
  server.focusOn(robot);
  double vr,vl,r=0.0335,R=0.1745,v,w;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Vector3d Euler, Init_Euler, bodyLinearVel_;
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
    v=1,w=0;
    vr = v/r + w*R/r;
    vl = v/r - w*R/r;
    vtarget4 << 0,vl, 0,vr, vl, vr;
    ptarget4 += vtarget4*world.getTimeStep();
    ptarget4[0] = 0.5, ptarget4[2] = 0.5;
    // std::cout<<ptarget.size()<<std::endl;
    ptarget.tail(nJoints) = ptarget4;
    vtarget.tail(nJoints) = vtarget4;
    robot->setPdTarget(ptarget,vtarget);
    
    // gc = robot->getGeneralizedCoordinate().e();
    // robot->getState(gc, gv);
    // Eigen::Quaterniond quaternion(gc[3], gc[4], gc[5], gc[6]);
    // Euler = quaternion.toRotationMatrix().eulerAngles(2, 1, 0);
    // raisim::Vec<4> quat;
    // raisim::Mat<3,3> rot;
    // quat[0] = gc[3]; quat[1] = gc[4]; quat[2] = gc[5]; quat[3] = gc[6];
    // raisim::quatToRotMat(quat, rot);
    // bodyLinearVel_ = rot.e().transpose() * gv.segment(0, 3);
    // std::cout<<"gc:"<<gc[3]<<" "<<gc[4]<<" "<<gc[5]<<" "<<gc[6]<<std::endl;
    // std::cout<<Euler[0]<<" "<<Euler[1]<<std::endl<<std::endl;
    // std::cout<<rot.e().row(2)<<std::endl;
    // std::cout<<bodyLinearVel_<<std::endl;
    if(fabs(gc[0])>35. || fabs(gc[1])>35.) {
      gc.segment<7>(0) << 0, 0, 2, 1, 0, 0, 0;
      gv.setRandom();
      robot->setState(gc, gv);
    }
  }

  server.killServer();
}

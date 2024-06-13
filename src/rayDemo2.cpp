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

  auto hm = world.addHeightMap("../data/height3.png", 0, 0, 20, 20, 0.00002, 0);
  // auto hm = world.addHeightMap("height2.png", 0, 0, 50, 50, 0.00005, 1.5);
  auto robot = world.addArticulatedSystem(binaryPath.getDirectory() + "../data/husky/husky.urdf");
  robot->setName("husky");
  hm->setAppearance("soil2");
  Eigen::VectorXd gc(robot->getGeneralizedCoordinateDim()), gv(robot->getDOF()), damping(robot->getDOF());
  gc.setZero(); gv.setZero();
  gc.segment<7>(0) << 5, 7, 0.2, 1, 0, 0, 0;
  robot->setGeneralizedCoordinate(gc);
  robot->setGeneralizedVelocity(gv);
  damping.setConstant(0);
  damping.tail(4).setConstant(1.);
  robot->setJointDamping(damping);

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

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  Eigen::Vector3d direction;

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
    robot->setGeneralizedForce({0, 0, 0, 0, 0, 0, -20, -20, -20, -20});
    gc = robot->getGeneralizedCoordinate().e();

    if(fabs(gc[0])>35. || fabs(gc[1])>35.) {
      gc.segment<7>(0) << 0, 0, 2, 1, 0, 0, 0;
      gv.setRandom();
      robot->setState(gc, gv);
    }
  }

  server.killServer();
}

#ifndef LIDAR_HPP
#define LIDAR_HPP

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include<pcl/io/ply_io.h>
#include <pcl/common/common_headers.h>
#include <iostream>

class lidar
{
private:
    raisim::Vec<3> lidarPos; 
    raisim::Mat<3,3> lidarOri;
    Eigen::Vector3d direction;
    Eigen::Vector3d rayDirection;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    int yawsize_,pitchsize_;
    bool visualizable_;
    double yaw;
    double pitch;
    raisim::InstancedVisuals* scans;
public:
    void init(int yawsize, int pitchsize, bool visualizable, raisim::RaisimServer& server){
        yawsize_ = yawsize;
        pitchsize_ = pitchsize;
        visualizable_ = visualizable;
        if(visualizable_){
            scans = server.addInstancedVisuals("scan points",
                                          raisim::Shape::Box,
                                          {0.01, 0.01, 0.01},
                                          {1,0,0,1},
                                          {0,1,0,1});
            scans->resize(yawsize_*pitchsize_);
        }

    };
    void scan(raisim::World& world, raisim::RaisimServer& server, raisim::ArticulatedSystem* robot);
    lidar();
    ~lidar();
};

lidar::lidar()
{
    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

lidar::~lidar()
{
}

void lidar::scan(raisim::World& world, raisim::RaisimServer& server, raisim::ArticulatedSystem* robot)
{
    cloud->clear();
    for(int i=0; i<yawsize_; i++) {
        for (int j = 0; j < pitchsize_; j++) {
            yaw = j * M_PI / pitchsize_ * 2 -  M_PI;
            pitch = -(i * 0.4/pitchsize_) + 0.25;
            const double normInv = 1. / sqrt(pitch * pitch + 1);
            direction = {cos(yaw) * normInv, sin(yaw) * normInv, -pitch * normInv};
            robot->getFramePosition("laser_joint", lidarPos);
            robot->getFrameOrientation("laser_joint", lidarOri);
            rayDirection = lidarOri.e() * direction;
            auto &col = world.rayTest(lidarPos.e(), rayDirection, 30);
            if (col.size() > 0) {
                auto relative_pos = col[0].getPosition() - lidarPos.e();
                float length = relative_pos.norm();
                pcl::PointXYZ point ={(float)relative_pos[0],(float)relative_pos[1],(float)(relative_pos[2]+0.5)};
                cloud->push_back(point);
                if(length>0){
                    if(visualizable_){
                        // std::cout<<"visualize"<<std::endl;
                        scans->setPosition(i * pitchsize_ + j, col[0].getPosition());
                        scans->setColorWeight(i * pitchsize_ + j, std::min(length/15.f, 1.0f));
                    }
                }
                else if(visualizable_)
                    scans->setPosition(i*pitchsize_+j, {0, 0, 100});
            }
            else if(visualizable_)
                scans->setPosition(i*pitchsize_+j, {0, 0, 100});
        }
    }
}

#endif // LIDAR_HPP
#ifndef LIDAR_HPP
#define LIDAR_HPP

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include<pcl/io/ply_io.h>
#include <pcl/common/common_headers.h>
#include <iostream>
#include "elevationMap.hpp"

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
public:
    elevation e_;
    void init(int yawsize, int pitchsize, bool visualizable){
        yawsize_ = yawsize;
        pitchsize_ = pitchsize;
        visualizable_ = visualizable;
    };
    void scan(std::unique_ptr<raisim::World>& world, std::unique_ptr<raisim::RaisimServer>& server, raisim::ArticulatedSystem* robot);
    void visualize(raisim::InstancedVisuals* scans);
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud(){return cloud;};
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

inline void lidar::scan(std::unique_ptr<raisim::World>& world, std::unique_ptr<raisim::RaisimServer>& server, raisim::ArticulatedSystem* robot)
{
    
    cloud->points.clear();
    for(int i=0; i<yawsize_; i++) {
        for (int j = 0; j < pitchsize_; j++) {
            yaw = j * M_PI / pitchsize_ * 2 -  M_PI;
            pitch = -(i * 0.4/pitchsize_) + 0.25;
            const double normInv = 1. / sqrt(pitch * pitch + 1);
            direction = {cos(yaw) * normInv, sin(yaw) * normInv, -pitch * normInv};
            robot->getFramePosition("imu_joint", lidarPos);
            robot->getFrameOrientation("imu_joint", lidarOri);
            rayDirection = lidarOri.e() * direction;
            auto &col = world->rayTest(lidarPos.e(), rayDirection, 30);
            if (col.size() > 0) {
                auto relative_pos = col[0].getPosition() - lidarPos.e();
                float length = relative_pos.norm();
                
                if(length>1){
                    pcl::PointXYZ point ={relative_pos[0],relative_pos[1],relative_pos[2]+0.5};
                    cloud->push_back(point);
                }
            }
        }
    }
    e_.Input(cloud);
}


inline void lidar::visualize(raisim::InstancedVisuals *scans)
{
    for(int i=0; i<cloud->points.size(); i++){
        scans->setPosition(i,{cloud->points[i].x, cloud->points[i].y, cloud->points[i].z});
    }
}

#endif // LIDAR_HPP
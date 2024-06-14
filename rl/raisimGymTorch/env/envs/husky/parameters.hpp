#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

class Parameters{
public:
    std::string map_path;
    std::string robot_urdf;
    double map_param[4];
    double gc_init[7];
    double goalpos[2];
    int scanSize[2];
    void update(const Yaml::Node& cfg){
        READ_YAML(std::string, map_path, cfg["map"]["path"]);
        READ_YAML(double, map_param[0], cfg["map"]["width"]);
        READ_YAML(double, map_param[1], cfg["map"]["height"]);
        READ_YAML(double, map_param[2], cfg["map"]["ratio"]);
        READ_YAML(double, map_param[3], cfg["map"]["z_offset"]);

        READ_YAML(std::string, robot_urdf, cfg["robot_urdf"]);

        READ_YAML(double, gc_init[0], cfg["gc_init"]["x"]);
        READ_YAML(double, gc_init[1], cfg["gc_init"]["y"]);
        READ_YAML(double, gc_init[2], cfg["gc_init"]["z"]);
        READ_YAML(double, gc_init[3], cfg["gc_init"]["qx"]);
        READ_YAML(double, gc_init[4], cfg["gc_init"]["qy"]);
        READ_YAML(double, gc_init[5], cfg["gc_init"]["qz"]);
        READ_YAML(double, gc_init[6], cfg["gc_init"]["qw"]);
        
        READ_YAML(double, goalpos[0], cfg["goal"]["x"]);
        READ_YAML(double, goalpos[1], cfg["goal"]["y"]);

        READ_YAML(int, scanSize[0], cfg["scan"]["size1"]);
        READ_YAML(int, scanSize[1], cfg["scan"]["size2"]);
        
    }
};
#endif
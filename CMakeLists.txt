cmake_minimum_required(VERSION 3.5)
project(3DNav LANGUAGES CXX)

#==
# Dependencies
#==
set(CMAKE_PREFIX_PATH "../../raisimLib/raisim/linux")
find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})
find_package(OpenCV 4.2 REQUIRED)
find_package(PCL 1.10 REQUIRED)

include_directories ( ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

#==
# Build Targets
#==
  
function(create_executable app_name file_name)
    add_executable(${app_name} ${file_name})

    target_link_libraries(${app_name} PUBLIC raisim::raisim pthread ${PCL_LIBRARIES} ${OpenCV_LIBS})
    target_include_directories(${app_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endfunction()

create_executable(rayDemo2 src/rayDemo2.cpp)
create_executable(elevationMap src/elevationMap.cpp)



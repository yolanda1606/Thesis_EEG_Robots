cmake_minimum_required(VERSION 3.4)
project(pick_and_place_cpp)

# Revert back to C++14
set(CMAKE_CXX_STANDARD 14)

# THE MAGIC LINE: Tell CMake to use our custom driver, NOT the ROS 2 one!
set(Franka_DIR "/home/sysgen/Projects/Yolanda/libfranka/build")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Franka REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Pick and Place
add_executable(pick_place src/pick_and_place.cpp)
target_link_libraries(pick_place Franka::Franka Eigen3::Eigen)

# Pose reader
add_executable(read_pose src/read_pose.cpp)
target_link_libraries(read_pose Franka::Franka)

# Simple Shape Shorter
add_executable(shape_sorter_obs src/simple_ss.cpp)
target_link_libraries(shape_sorter_obs Franka::Franka Eigen3::Eigen)

# HRI Shape Sorter
add_executable(shape_sorter_hri src/hri_ss.cpp)
target_link_libraries(shape_sorter_hri Franka::Franka Eigen3::Eigen)

# Sisyphus
add_executable(sisyphus_task src/sisyphus.cpp)
target_link_libraries(sisyphus_task Franka::Franka Eigen3::Eigen Threads::Threads)

# Stack
add_executable(stack src/stack.cpp)
target_link_libraries(stack Franka::Franka Eigen3::Eigen)

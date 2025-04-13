cmake_minimum_required(VERSION 3.13)

project(objdet_node VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(Boost COMPONENTS date_time)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
#find_package(visualization_msgs REQUIRED)
#find_package(vision_msgs)
#find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_executable(objdet_exe "${CMAKE_CURRENT_SOURCE_DIR}/src/ObjDet.cpp")

target_link_libraries(objdet_exe ${PCL_LIBRARIES})

#target_include_directories(objdet_exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(objdet_exe PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(objdet_exe rclcpp tf2_ros tf2 tf2_eigen sensor_msgs pcl_ros)#rclcpp sensor_msgs pcl_ros)

install(TARGETS objdet_exe
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_package()

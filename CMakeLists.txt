cmake_minimum_required(VERSION 3.8)
project(ground_segmentation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(PCL REQUIRED COMPONENTS common io visualization filters)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions  REQUIRED)


find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

ament_auto_find_build_dependencies()

include_directories(
        include
        SYSTEM
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}        
        ${PCL_INCLUDE_DIRS}
)

ament_auto_add_library(ground_segmentation SHARED
        src/travel_ground_filter_nodelet.cpp
        src/travel_awf_node.cpp
)

if(${PCL_VERSION} GREATER_EQUAL 1.12.1)
    find_package(Qhull REQUIRED)
    target_link_libraries(ground_segmentation
            QHULL::QHULL
            )
endif()

target_link_libraries(ground_segmentation
        Eigen3::Eigen
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PCL_LIBRARIES}
        )

rclcpp_components_register_node(ground_segmentation
        PLUGIN "ground_segmentation::Travel"
        EXECUTABLE travel_node
)

ament_auto_package()

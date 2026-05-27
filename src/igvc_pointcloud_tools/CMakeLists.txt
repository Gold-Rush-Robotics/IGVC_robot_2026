cmake_minimum_required(VERSION 3.8)
project(igvc_pointcloud_tools)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_library(pointcloud_merger_component SHARED
  src/pointcloud_merger_node.cpp
)
target_compile_features(pointcloud_merger_component PUBLIC cxx_std_17)
target_include_directories(pointcloud_merger_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(pointcloud_merger_component
  rclcpp
  rclcpp_components
  sensor_msgs
  tf2
  tf2_ros
)
rclcpp_components_register_nodes(pointcloud_merger_component
  "igvc_pointcloud_tools::PointCloudMergerNode")

add_executable(pointcloud_merger_node
  src/pointcloud_merger_main.cpp
)
target_compile_features(pointcloud_merger_node PUBLIC cxx_std_17)
target_link_libraries(pointcloud_merger_node pointcloud_merger_component)
ament_target_dependencies(pointcloud_merger_node rclcpp)

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS pointcloud_merger_component pointcloud_merger_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  rclcpp_components
  sensor_msgs
  tf2
  tf2_ros
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

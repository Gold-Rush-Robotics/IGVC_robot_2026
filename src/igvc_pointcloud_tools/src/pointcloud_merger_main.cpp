#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "igvc_pointcloud_tools/pointcloud_merger_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);

  auto node = std::make_shared<igvc_pointcloud_tools::PointCloudMergerNode>(options);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}

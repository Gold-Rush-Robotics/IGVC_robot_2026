#ifndef IGVC_POINTCLOUD_TOOLS__POINTCLOUD_MERGER_NODE_HPP_
#define IGVC_POINTCLOUD_TOOLS__POINTCLOUD_MERGER_NODE_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace igvc_pointcloud_tools
{

class PointCloudMergerNode final : public rclcpp::Node
{
public:
  explicit PointCloudMergerNode(const rclcpp::NodeOptions & options);

private:
  struct CachedCloud
  {
    sensor_msgs::msg::PointCloud2::UniquePtr msg;
    rclcpp::Time receive_time;
  };

  struct FieldOffsets
  {
    uint32_t x{0};
    uint32_t y{0};
    uint32_t z{0};
  };

  struct Transform3
  {
    bool identity{true};
    double r00{1.0};
    double r01{0.0};
    double r02{0.0};
    double r10{0.0};
    double r11{1.0};
    double r12{0.0};
    double r20{0.0};
    double r21{0.0};
    double r22{1.0};
    double tx{0.0};
    double ty{0.0};
    double tz{0.0};
  };

  void on_cloud(std::size_t index, sensor_msgs::msg::PointCloud2::UniquePtr msg);
  void maybe_publish_locked();

  bool find_xyz_fields(
    const sensor_msgs::msg::PointCloud2 & cloud,
    FieldOffsets & offsets) const;
  bool same_layout(
    const sensor_msgs::msg::PointCloud2 & lhs,
    const sensor_msgs::msg::PointCloud2 & rhs) const;
  bool lookup_transform(
    const sensor_msgs::msg::PointCloud2 & cloud,
    Transform3 & transform) const;
  void append_cloud(
    const sensor_msgs::msg::PointCloud2 & input,
    const Transform3 & transform,
    const FieldOffsets & offsets,
    sensor_msgs::msg::PointCloud2 & output,
    std::size_t & appended_points) const;

  std::vector<std::string> input_topics_;
  std::string output_topic_;
  std::string target_frame_;
  double max_stamp_delta_sec_{0.25};
  double transform_timeout_sec_{0.05};
  double publish_hz_{15.0};
  bool drop_invalid_{false};
  bool preserve_organized_{true};

  rclcpp::QoS qos_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

  std::vector<CachedCloud> latest_;
  rclcpp::Time last_publish_time_;
  mutable std::mutex mutex_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace igvc_pointcloud_tools

#endif  // IGVC_POINTCLOUD_TOOLS__POINTCLOUD_MERGER_NODE_HPP_

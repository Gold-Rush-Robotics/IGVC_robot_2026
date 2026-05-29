#ifndef IGVC_POINTCLOUD_TOOLS__POINTCLOUD_MERGER_NODE_HPP_
#define IGVC_POINTCLOUD_TOOLS__POINTCLOUD_MERGER_NODE_HPP_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
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

  using CalibCloud = pcl::PointCloud<pcl::PointXYZ>;

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

  // ── One-shot geometric calibration helpers ──────────────────────
  static Eigen::Matrix4f transform3_to_eigen(const Transform3 & t);
  static Transform3 eigen_to_transform3(const Eigen::Matrix4f & m);
  void msg_to_pcl_xyz(
    const sensor_msgs::msg::PointCloud2 & msg,
    const FieldOffsets & offsets,
    CalibCloud & out) const;
  void accumulate_for_calibration_locked(
    std::size_t index,
    const sensor_msgs::msg::PointCloud2 & cloud,
    const FieldOffsets & offsets,
    const Transform3 & urdf_in_base);
  bool ready_to_calibrate_locked() const;
  void run_calibration_locked(const std::vector<Transform3> & urdf_in_base);
  bool run_icp(
    const CalibCloud & source_in_base,
    const CalibCloud & target_in_base,
    Eigen::Matrix4f & refinement,
    double & fitness) const;

  std::vector<std::string> input_topics_;
  std::string output_topic_;
  std::string target_frame_;
  double max_stamp_delta_sec_{0.25};
  double transform_timeout_sec_{0.05};
  double publish_hz_{15.0};
  bool drop_invalid_{false};
  bool preserve_organized_{true};

  // ── Geometric (ICP/FPFH) calibration parameters ─────────────────
  // When enabled, the merger accumulates calibration_frames_per_input_
  // frames per input (downsampled to base_link via URDF), then runs
  // ICP from each non-reference input against the reference input.
  // The resulting refined source→base_link transform is cached and
  // reused for every subsequent frame; URDF is no longer consulted.
  bool calibrate_on_startup_{false};
  std::string calibration_method_{"icp"};   // "icp" (only one for now)
  std::size_t calibration_reference_index_{0};
  int calibration_frames_per_input_{10};
  double calibration_voxel_size_m_{0.10};
  double calibration_max_correspondence_m_{0.30};
  int calibration_max_iterations_{50};
  std::size_t calibration_max_points_per_input_{200000};
  // Sanity gates: if ICP refinement exceeds these or fitness is worse
  // than the threshold, fall back to URDF for that camera.
  double calibration_max_translation_m_{0.30};
  double calibration_max_rotation_deg_{15.0};
  double calibration_max_fitness_{0.10};

  rclcpp::QoS qos_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

  std::vector<CachedCloud> latest_;
  rclcpp::Time last_publish_time_;
  mutable std::mutex mutex_;

  // Calibration runtime state.  All writes under mutex_; calibration_done_
  // is also atomic so lookup_transform() can short-circuit without locking.
  std::atomic<bool> calibration_done_{false};
  std::vector<std::shared_ptr<CalibCloud>> calibration_accum_;
  std::vector<int> calibration_frame_counts_;
  mutable std::unordered_map<std::string, Transform3> calibrated_transforms_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace igvc_pointcloud_tools

#endif  // IGVC_POINTCLOUD_TOOLS__POINTCLOUD_MERGER_NODE_HPP_

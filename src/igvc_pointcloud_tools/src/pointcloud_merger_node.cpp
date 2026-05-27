#include "igvc_pointcloud_tools/pointcloud_merger_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>

namespace igvc_pointcloud_tools
{
namespace
{
constexpr const char * kDefaultFrontCloud =
  "/front_zed_camera_x/zed_node/point_cloud/cloud_registered";
constexpr const char * kDefaultLeftCloud =
  "/left_zed_camera_x/zed_node/point_cloud/cloud_registered";
constexpr const char * kDefaultRightCloud =
  "/right_zed_camera_x/zed_node/point_cloud/cloud_registered";

float read_float32(const std::uint8_t * ptr)
{
  float value = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(&value, ptr, sizeof(float));
  return value;
}

void write_float32(std::uint8_t * ptr, float value)
{
  std::memcpy(ptr, &value, sizeof(float));
}
}  // namespace

PointCloudMergerNode::PointCloudMergerNode(const rclcpp::NodeOptions & options)
: Node("pointcloud_merger_node", options),
  qos_(rclcpp::SensorDataQoS()),
  last_publish_time_(0, 0, this->get_clock()->get_clock_type())
{
  input_topics_ = this->declare_parameter<std::vector<std::string>>(
    "input_topics", {kDefaultFrontCloud, kDefaultLeftCloud, kDefaultRightCloud});
  output_topic_ = this->declare_parameter<std::string>(
    "output_topic", "/zed/point_cloud/merged_registered");
  target_frame_ = this->declare_parameter<std::string>("target_frame", "base_link");
  max_stamp_delta_sec_ = this->declare_parameter<double>("max_stamp_delta_sec", 0.25);
  transform_timeout_sec_ = this->declare_parameter<double>("transform_timeout_sec", 0.05);
  publish_hz_ = this->declare_parameter<double>("publish_hz", 15.0);
  drop_invalid_ = this->declare_parameter<bool>("drop_invalid", false);
  preserve_organized_ = this->declare_parameter<bool>("preserve_organized", true);

  if (input_topics_.empty()) {
    throw std::runtime_error("pointcloud_merger_node requires at least one input topic");
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_buffer_->setUsingDedicatedThread(true);
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, true);

  pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, qos_);
  latest_.resize(input_topics_.size());
  subs_.reserve(input_topics_.size());

  for (std::size_t i = 0; i < input_topics_.size(); ++i) {
    subs_.push_back(this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topics_[i], qos_,
      [this, i](sensor_msgs::msg::PointCloud2::UniquePtr msg) {
        this->on_cloud(i, std::move(msg));
      }));
    RCLCPP_INFO(this->get_logger(), "pointcloud_merger input[%zu]=%s", i, input_topics_[i].c_str());
  }

  RCLCPP_INFO(
    this->get_logger(),
    "pointcloud_merger publishing %zu clouds -> %s in frame %s "
    "(intra_process=%s, preserve_organized=%s)",
    input_topics_.size(), output_topic_.c_str(), target_frame_.c_str(),
    this->get_node_options().use_intra_process_comms() ? "true" : "false",
    preserve_organized_ ? "true" : "false");
}

void PointCloudMergerNode::on_cloud(
  std::size_t index, sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_[index].receive_time = this->now();
  latest_[index].msg = std::move(msg);
  maybe_publish_locked();
}

bool PointCloudMergerNode::find_xyz_fields(
  const sensor_msgs::msg::PointCloud2 & cloud,
  FieldOffsets & offsets) const
{
  bool has_x = false;
  bool has_y = false;
  bool has_z = false;
  for (const auto & field : cloud.fields) {
    if (field.datatype != sensor_msgs::msg::PointField::FLOAT32 || field.count != 1) {
      continue;
    }
    if (field.name == "x") {
      offsets.x = field.offset;
      has_x = true;
    } else if (field.name == "y") {
      offsets.y = field.offset;
      has_y = true;
    } else if (field.name == "z") {
      offsets.z = field.offset;
      has_z = true;
    }
  }
  return has_x && has_y && has_z;
}

bool PointCloudMergerNode::same_layout(
  const sensor_msgs::msg::PointCloud2 & lhs,
  const sensor_msgs::msg::PointCloud2 & rhs) const
{
  if (
    lhs.is_bigendian != rhs.is_bigendian ||
    lhs.point_step != rhs.point_step ||
    lhs.fields.size() != rhs.fields.size())
  {
    return false;
  }

  for (std::size_t i = 0; i < lhs.fields.size(); ++i) {
    const auto & a = lhs.fields[i];
    const auto & b = rhs.fields[i];
    if (
      a.name != b.name ||
      a.offset != b.offset ||
      a.datatype != b.datatype ||
      a.count != b.count)
    {
      return false;
    }
  }
  return true;
}

bool PointCloudMergerNode::lookup_transform(
  const sensor_msgs::msg::PointCloud2 & cloud,
  Transform3 & transform) const
{
  const std::string source_frame = cloud.header.frame_id;
  if (target_frame_.empty() || source_frame.empty() || source_frame == target_frame_) {
    transform = Transform3{};
    transform.identity = true;
    return true;
  }

  try {
    const auto stamped = tf_buffer_->lookupTransform(
      target_frame_, source_frame, tf2::TimePointZero,
      tf2::durationFromSec(transform_timeout_sec_));
    const auto & q = stamped.transform.rotation;
    const auto & t = stamped.transform.translation;

    transform.identity = false;
    transform.r00 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    transform.r01 = 2.0 * (q.x * q.y - q.z * q.w);
    transform.r02 = 2.0 * (q.x * q.z + q.y * q.w);
    transform.r10 = 2.0 * (q.x * q.y + q.z * q.w);
    transform.r11 = 1.0 - 2.0 * (q.x * q.x + q.z * q.z);
    transform.r12 = 2.0 * (q.y * q.z - q.x * q.w);
    transform.r20 = 2.0 * (q.x * q.z - q.y * q.w);
    transform.r21 = 2.0 * (q.y * q.z + q.x * q.w);
    transform.r22 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    transform.tx = t.x;
    transform.ty = t.y;
    transform.tz = t.z;
    return true;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Waiting for TF %s <- %s before merging clouds: %s",
      target_frame_.c_str(), source_frame.c_str(), ex.what());
    return false;
  }
}

void PointCloudMergerNode::append_cloud(
  const sensor_msgs::msg::PointCloud2 & input,
  const Transform3 & transform,
  const FieldOffsets & offsets,
  sensor_msgs::msg::PointCloud2 & output,
  std::size_t & appended_points) const
{
  for (std::uint32_t row = 0; row < input.height; ++row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * input.row_step;
    for (std::uint32_t col = 0; col < input.width; ++col) {
      const auto * src = input.data.data() + row_offset + static_cast<std::size_t>(col) * input.point_step;
      const float x = read_float32(src + offsets.x);
      const float y = read_float32(src + offsets.y);
      const float z = read_float32(src + offsets.z);
      const bool finite = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
      if (drop_invalid_ && !finite) {
        continue;
      }

      const std::size_t dst_offset = output.data.size();
      output.data.resize(dst_offset + input.point_step);
      auto * dst = output.data.data() + dst_offset;
      std::memcpy(dst, src, input.point_step);

      if (!transform.identity && finite) {
        const double bx = transform.r00 * x + transform.r01 * y + transform.r02 * z + transform.tx;
        const double by = transform.r10 * x + transform.r11 * y + transform.r12 * z + transform.ty;
        const double bz = transform.r20 * x + transform.r21 * y + transform.r22 * z + transform.tz;
        write_float32(dst + offsets.x, static_cast<float>(bx));
        write_float32(dst + offsets.y, static_cast<float>(by));
        write_float32(dst + offsets.z, static_cast<float>(bz));
      }
      ++appended_points;
    }
  }
}

void PointCloudMergerNode::maybe_publish_locked()
{
  for (const auto & cached : latest_) {
    if (!cached.msg) {
      return;
    }
  }

  const auto now = this->now();
  if (publish_hz_ > 0.0) {
    const auto min_period = rclcpp::Duration::from_seconds(1.0 / publish_hz_);
    if ((now - last_publish_time_) < min_period) {
      return;
    }
  }

  rclcpp::Time min_stamp(latest_[0].msg->header.stamp);
  rclcpp::Time max_stamp(latest_[0].msg->header.stamp);
  for (const auto & cached : latest_) {
    const rclcpp::Time stamp(cached.msg->header.stamp);
    if (stamp < min_stamp) {
      min_stamp = stamp;
    }
    if (stamp > max_stamp) {
      max_stamp = stamp;
    }
  }
  if (max_stamp_delta_sec_ > 0.0 && (max_stamp - min_stamp).seconds() > max_stamp_delta_sec_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Skipping point cloud merge: input stamp spread %.3f s > %.3f s",
      (max_stamp - min_stamp).seconds(), max_stamp_delta_sec_);
    return;
  }

  const auto & reference = *latest_[0].msg;
  if (reference.is_bigendian) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Skipping point cloud merge: big-endian PointCloud2 is not supported");
    return;
  }

  FieldOffsets offsets;
  if (!find_xyz_fields(reference, offsets)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Skipping point cloud merge: reference cloud has no float32 x/y/z fields");
    return;
  }

  std::vector<Transform3> transforms(latest_.size());
  bool organized = preserve_organized_ && !drop_invalid_ && reference.width > 0;
  std::uint32_t organized_width = reference.width;
  std::uint32_t organized_height = 0;
  std::size_t reserve_bytes = 0;

  for (std::size_t i = 0; i < latest_.size(); ++i) {
    const auto & cloud = *latest_[i].msg;
    if (cloud.is_bigendian || !same_layout(reference, cloud)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Skipping point cloud merge: input[%zu] layout differs from input[0]", i);
      return;
    }
    if (organized && cloud.width != organized_width) {
      organized = false;
    }
    organized_height += cloud.height;
    reserve_bytes += static_cast<std::size_t>(cloud.width) * cloud.height * cloud.point_step;
    if (!lookup_transform(cloud, transforms[i])) {
      return;
    }
  }

  auto output = std::make_unique<sensor_msgs::msg::PointCloud2>();
  output->header.stamp = max_stamp;
  output->header.frame_id = target_frame_.empty() ? reference.header.frame_id : target_frame_;
  output->fields = reference.fields;
  output->is_bigendian = reference.is_bigendian;
  output->point_step = reference.point_step;
  output->is_dense = false;
  output->data.reserve(reserve_bytes);

  std::size_t appended_points = 0;
  for (std::size_t i = 0; i < latest_.size(); ++i) {
    append_cloud(*latest_[i].msg, transforms[i], offsets, *output, appended_points);
  }

  if (organized) {
    output->height = organized_height;
    output->width = organized_width;
  } else {
    output->height = 1;
    output->width = static_cast<std::uint32_t>(appended_points);
  }
  output->row_step = output->point_step * output->width;

  last_publish_time_ = now;
  pub_->publish(std::move(output));
}

}  // namespace igvc_pointcloud_tools

RCLCPP_COMPONENTS_REGISTER_NODE(igvc_pointcloud_tools::PointCloudMergerNode)

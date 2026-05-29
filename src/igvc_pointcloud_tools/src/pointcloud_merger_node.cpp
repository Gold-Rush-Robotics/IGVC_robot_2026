#include "igvc_pointcloud_tools/pointcloud_merger_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
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

  // ── Geometric (ICP) calibration ─────────────────────────────────
  calibrate_on_startup_ = this->declare_parameter<bool>(
    "calibrate_on_startup", false);
  calibration_method_ = this->declare_parameter<std::string>(
    "calibration_method", std::string("icp"));
  calibration_reference_index_ = static_cast<std::size_t>(std::max<int>(
    0, this->declare_parameter<int>("calibration_reference_index", 0)));
  calibration_frames_per_input_ = std::max<int>(
    1, this->declare_parameter<int>("calibration_frames_per_input", 10));
  calibration_voxel_size_m_ = std::max<double>(
    1e-3, this->declare_parameter<double>("calibration_voxel_size_m", 0.10));
  calibration_max_correspondence_m_ = std::max<double>(
    1e-3, this->declare_parameter<double>("calibration_max_correspondence_m", 0.30));
  calibration_max_iterations_ = std::max<int>(
    1, this->declare_parameter<int>("calibration_max_iterations", 50));
  calibration_max_points_per_input_ = static_cast<std::size_t>(std::max<int>(
    1, this->declare_parameter<int>("calibration_max_points_per_input", 200000)));
  calibration_max_translation_m_ = std::max<double>(
    0.0, this->declare_parameter<double>("calibration_max_translation_m", 0.30));
  calibration_max_rotation_deg_ = std::max<double>(
    0.0, this->declare_parameter<double>("calibration_max_rotation_deg", 15.0));
  calibration_max_fitness_ = std::max<double>(
    0.0, this->declare_parameter<double>("calibration_max_fitness", 0.10));

  if (calibration_method_ != "icp") {
    RCLCPP_WARN(
      this->get_logger(),
      "calibration_method='%s' not supported (only 'icp' is implemented); "
      "forcing 'icp'.", calibration_method_.c_str());
    calibration_method_ = "icp";
  }

  if (input_topics_.empty()) {
    throw std::runtime_error("pointcloud_merger_node requires at least one input topic");
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_buffer_->setUsingDedicatedThread(true);
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, true);

  pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, qos_);
  latest_.resize(input_topics_.size());
  subs_.reserve(input_topics_.size());
  calibration_accum_.resize(input_topics_.size());
  calibration_frame_counts_.assign(input_topics_.size(), 0);
  for (auto & cloud_ptr : calibration_accum_) {
    cloud_ptr = std::make_shared<CalibCloud>();
  }
  if (calibrate_on_startup_ &&
      calibration_reference_index_ >= input_topics_.size())
  {
    RCLCPP_WARN(
      this->get_logger(),
      "calibration_reference_index=%zu out of range (%zu inputs); "
      "using 0 as reference.",
      calibration_reference_index_, input_topics_.size());
    calibration_reference_index_ = 0;
  }

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
    "(intra_process=%s, preserve_organized=%s, calibrate_on_startup=%s, "
    "calibration_method=%s, reference_index=%zu)",
    input_topics_.size(), output_topic_.c_str(), target_frame_.c_str(),
    this->get_node_options().use_intra_process_comms() ? "true" : "false",
    preserve_organized_ ? "true" : "false",
    calibrate_on_startup_ ? "true" : "false",
    calibration_method_.c_str(),
    calibration_reference_index_);
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

  // Calibrated mode: once geometric calibration has finished, the
  // cached per-source-frame transform is used forever and URDF/TF
  // updates are ignored.
  if (calibration_done_.load(std::memory_order_acquire)) {
    const auto it = calibrated_transforms_.find(source_frame);
    if (it != calibrated_transforms_.end()) {
      transform = it->second;
      return true;
    }
    // Fall through to URDF lookup if this source frame was not part
    // of the calibration set (e.g. a new topic appeared at runtime).
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

// ── One-shot geometric calibration ──────────────────────────────

Eigen::Matrix4f PointCloudMergerNode::transform3_to_eigen(const Transform3 & t)
{
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  if (t.identity) {
    return m;
  }
  m(0, 0) = static_cast<float>(t.r00);
  m(0, 1) = static_cast<float>(t.r01);
  m(0, 2) = static_cast<float>(t.r02);
  m(0, 3) = static_cast<float>(t.tx);
  m(1, 0) = static_cast<float>(t.r10);
  m(1, 1) = static_cast<float>(t.r11);
  m(1, 2) = static_cast<float>(t.r12);
  m(1, 3) = static_cast<float>(t.ty);
  m(2, 0) = static_cast<float>(t.r20);
  m(2, 1) = static_cast<float>(t.r21);
  m(2, 2) = static_cast<float>(t.r22);
  m(2, 3) = static_cast<float>(t.tz);
  return m;
}

PointCloudMergerNode::Transform3 PointCloudMergerNode::eigen_to_transform3(
  const Eigen::Matrix4f & m)
{
  Transform3 t;
  t.identity = false;
  t.r00 = m(0, 0); t.r01 = m(0, 1); t.r02 = m(0, 2); t.tx = m(0, 3);
  t.r10 = m(1, 0); t.r11 = m(1, 1); t.r12 = m(1, 2); t.ty = m(1, 3);
  t.r20 = m(2, 0); t.r21 = m(2, 1); t.r22 = m(2, 2); t.tz = m(2, 3);
  return t;
}

void PointCloudMergerNode::msg_to_pcl_xyz(
  const sensor_msgs::msg::PointCloud2 & msg,
  const FieldOffsets & offsets,
  CalibCloud & out) const
{
  out.clear();
  const std::size_t total =
    static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
  out.reserve(total);
  for (std::uint32_t row = 0; row < msg.height; ++row) {
    const std::size_t row_off =
      static_cast<std::size_t>(row) * msg.row_step;
    for (std::uint32_t col = 0; col < msg.width; ++col) {
      const auto * p = msg.data.data() + row_off +
        static_cast<std::size_t>(col) * msg.point_step;
      const float x = read_float32(p + offsets.x);
      const float y = read_float32(p + offsets.y);
      const float z = read_float32(p + offsets.z);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }
      out.emplace_back(x, y, z);
    }
  }
  out.is_dense = true;
  out.width = static_cast<std::uint32_t>(out.size());
  out.height = 1;
}

void PointCloudMergerNode::accumulate_for_calibration_locked(
  std::size_t index,
  const sensor_msgs::msg::PointCloud2 & cloud,
  const FieldOffsets & offsets,
  const Transform3 & urdf_in_base)
{
  if (index >= calibration_accum_.size()) {
    return;
  }
  if (calibration_frame_counts_[index] >= calibration_frames_per_input_) {
    return;
  }
  if (calibration_accum_[index]->size() >= calibration_max_points_per_input_) {
    // Stop accumulating raw points but still count this frame as
    // observed so the readiness check can succeed.
    calibration_frame_counts_[index]++;
    return;
  }

  CalibCloud raw;
  msg_to_pcl_xyz(cloud, offsets, raw);
  if (raw.empty()) {
    return;
  }

  // Transform raw source-frame points into base_link via URDF.
  CalibCloud in_base;
  pcl::transformPointCloud(raw, in_base, transform3_to_eigen(urdf_in_base));

  // Voxel downsample to keep memory bounded.
  CalibCloud::Ptr in_base_ptr = in_base.makeShared();
  CalibCloud downsampled;
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  const auto leaf = static_cast<float>(calibration_voxel_size_m_);
  vg.setLeafSize(leaf, leaf, leaf);
  vg.setInputCloud(in_base_ptr);
  vg.filter(downsampled);

  auto & accum = *calibration_accum_[index];
  accum.insert(accum.end(), downsampled.begin(), downsampled.end());
  accum.width = static_cast<std::uint32_t>(accum.size());
  accum.height = 1;
  accum.is_dense = true;
  calibration_frame_counts_[index]++;
}

bool PointCloudMergerNode::ready_to_calibrate_locked() const
{
  for (int count : calibration_frame_counts_) {
    if (count < calibration_frames_per_input_) {
      return false;
    }
  }
  return true;
}

bool PointCloudMergerNode::run_icp(
  const CalibCloud & source_in_base,
  const CalibCloud & target_in_base,
  Eigen::Matrix4f & refinement,
  double & fitness) const
{
  if (source_in_base.empty() || target_in_base.empty()) {
    return false;
  }

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(source_in_base.makeShared());
  icp.setInputTarget(target_in_base.makeShared());
  icp.setMaxCorrespondenceDistance(calibration_max_correspondence_m_);
  icp.setMaximumIterations(calibration_max_iterations_);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);

  CalibCloud aligned;
  icp.align(aligned);
  if (!icp.hasConverged()) {
    return false;
  }
  refinement = icp.getFinalTransformation();
  fitness = icp.getFitnessScore();
  if (!std::isfinite(fitness) || !refinement.allFinite()) {
    return false;
  }
  return true;
}

void PointCloudMergerNode::run_calibration_locked(
  const std::vector<Transform3> & urdf_in_base)
{
  const std::size_t ref = calibration_reference_index_;
  if (ref >= calibration_accum_.size() || ref >= urdf_in_base.size()) {
    calibration_done_.store(true, std::memory_order_release);
    return;
  }

  const auto & target_cloud = *calibration_accum_[ref];
  RCLCPP_INFO(
    this->get_logger(),
    "Running ICP calibration: reference=%s (%zu pts), "
    "voxel=%.3f m, max_corr=%.3f m, max_iters=%d, fitness_cap=%.3f",
    input_topics_[ref].c_str(), target_cloud.size(),
    calibration_voxel_size_m_, calibration_max_correspondence_m_,
    calibration_max_iterations_, calibration_max_fitness_);

  for (std::size_t i = 0; i < calibration_accum_.size(); ++i) {
    const auto & source_frame = latest_[i].msg
      ? latest_[i].msg->header.frame_id
      : std::string{};
    Transform3 calibrated = urdf_in_base[i];  // default = URDF fallback

    if (i == ref) {
      RCLCPP_INFO(
        this->get_logger(),
        "  [%zu] %s reference camera: using URDF (no refinement).",
        i, source_frame.c_str());
    } else {
      Eigen::Matrix4f refine = Eigen::Matrix4f::Identity();
      double fitness = std::numeric_limits<double>::infinity();
      const bool ok = run_icp(
        *calibration_accum_[i], target_cloud, refine, fitness);
      const Eigen::Vector3f t_refine = refine.block<3, 1>(0, 3);
      const Eigen::Matrix3f R_refine = refine.block<3, 3>(0, 0);
      const float cos_angle = std::clamp(
        0.5f * (R_refine.trace() - 1.0f), -1.0f, 1.0f);
      const double angle_deg =
        static_cast<double>(std::acos(cos_angle)) * 180.0 / M_PI;
      const double t_norm = static_cast<double>(t_refine.norm());

      const bool refine_within_caps =
        ok &&
        fitness <= calibration_max_fitness_ &&
        t_norm <= calibration_max_translation_m_ &&
        angle_deg <= calibration_max_rotation_deg_;

      if (refine_within_caps) {
        // Calibrated source→base = refine * URDF_source→base.
        const Eigen::Matrix4f urdf_mat =
          transform3_to_eigen(urdf_in_base[i]);
        const Eigen::Matrix4f final_mat = refine * urdf_mat;
        calibrated = eigen_to_transform3(final_mat);
        RCLCPP_INFO(
          this->get_logger(),
          "  [%zu] %s ICP OK: dt=[%.4f,%.4f,%.4f] m (|dt|=%.4f), "
          "dtheta=%.3f deg, fitness=%.4f -> using REFINED transform.",
          i, source_frame.c_str(),
          t_refine.x(), t_refine.y(), t_refine.z(),
          t_norm, angle_deg, fitness);
      } else {
        RCLCPP_WARN(
          this->get_logger(),
          "  [%zu] %s ICP rejected (converged=%s, fitness=%.4f, "
          "|dt|=%.4f m, dtheta=%.3f deg) -> falling back to URDF.",
          i, source_frame.c_str(),
          ok ? "true" : "false",
          fitness, t_norm, angle_deg);
      }
    }

    if (!source_frame.empty()) {
      calibrated_transforms_[source_frame] = calibrated;
    }
  }

  // Free the heavy accumulation buffers; we no longer need them.
  for (auto & cloud_ptr : calibration_accum_) {
    cloud_ptr.reset();
  }
  calibration_accum_.clear();
  calibration_frame_counts_.clear();

  calibration_done_.store(true, std::memory_order_release);
  RCLCPP_INFO(this->get_logger(), "Geometric calibration complete.");
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

  // ── One-shot geometric calibration pass ─────────────────────────
  // While calibration is in progress, accumulate downsampled points
  // from each input (already transformed to base_link via URDF) and
  // skip publishing.  Once we have enough frames per input, run ICP
  // to refine each non-reference camera's transform; after that, all
  // future frames flow through the cached calibrated transforms.
  if (calibrate_on_startup_ &&
      !calibration_done_.load(std::memory_order_acquire))
  {
    for (std::size_t i = 0; i < latest_.size(); ++i) {
      accumulate_for_calibration_locked(
        i, *latest_[i].msg, offsets, transforms[i]);
    }
    if (ready_to_calibrate_locked()) {
      run_calibration_locked(transforms);
      // Re-lookup transforms now that calibrated_transforms_ is
      // populated so this very frame publishes with the refined data.
      for (std::size_t i = 0; i < latest_.size(); ++i) {
        if (!lookup_transform(*latest_[i].msg, transforms[i])) {
          return;
        }
      }
    } else {
      // Still gathering frames; defer publishing.
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

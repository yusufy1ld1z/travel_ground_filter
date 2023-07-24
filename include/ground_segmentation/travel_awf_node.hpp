#ifndef AWF_TRAVEL_NODE_HPP
#define AWF_TRAVEL_NODE_HPP

#include "ground_segmentation/travel_ground_filter_nodelet.hpp"

#include <boost/shared_ptr.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>

namespace ground_segmentation {

class Travel : public rclcpp::Node {
public:
  explicit Travel(const rclcpp::NodeOptions &node_options);

private:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pc_cloud_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pc_ground_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pc_nonground_publisher_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      pc_cloud_subscriber_;

  void point_cloud_callback(
      const sensor_msgs::msg::PointCloud2::SharedPtr point_cloud);

  void show_parameters();

  std::shared_ptr<TravelGroundFilterComponent> TravelGroundFilterComponent_;

  // Patchwork parameters
  double max_range;
  double min_range;
  double resolution;
  int num_iter;
  int num_lpr;
  int num_min_pts;
  double th_seeds;
  double th_dist;
  double th_outlier;
  double th_normal;
  double th_weight;
  double th_lcc_normal_similiarity;
  double th_lcc_planar_model_dist;
  double th_obstacle;
  bool refine_mode;
  bool visualization;
};
} // namespace ground_segmentation
#endif // AWF_TRAVEL_NODE_HPP

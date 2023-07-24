#ifndef TRAVEL_GROUND_FILTER_NODELET_HPP
#define TRAVEL_GROUND_FILTER_NODELET_HPP

// include "pointcloud_preprocessor/filter.hpp"

#include <iostream>
#include <math.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <queue>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>

namespace ground_segmentation {

#define PTCLOUD_SIZE 132000 // may be changed
#define NODEWISE_PTCLOUDSIZE 5000

#define UNKNOWN 1 // maybe enum implementation
#define NONGROUND 2
#define GROUND 3

using PointType = pcl::PointXYZI;

using Eigen::JacobiSVD;
using Eigen::MatrixXf;
using Eigen::VectorXf;

struct TriGridIdx {
  int row, col, tri;
};

struct TriGridEdge {
  std::pair<TriGridIdx, TriGridIdx> Pair;
  bool is_traversable;
};

struct TriGridNode {
  int node_type;
  pcl::PointCloud<PointType> ptCloud;

  bool is_curr_data;

  // planar model
  Eigen::Vector3f normal;
  Eigen::Vector3f mean_pt;
  double d;

  Eigen::Vector3f singular_values;
  Eigen::Matrix3f eigen_vectors;
  double weight;

  double th_dist_d;
  double th_outlier_d;

  // graph_searching
  bool need_recheck;
  bool is_visited;
  bool is_rejection;
  int check_life;
  int depth;
};

struct TriGridCorner {
  double x, y;
  std::vector<double> zs;
  std::vector<double> weights;
};

using GridNode = std::vector<TriGridNode>;
using TriGridField = std::vector<std::vector<GridNode>>;

class TravelGroundFilterComponent : public rclcpp::Node {
private:
  pcl::PCLHeader cloud_header_;
  std_msgs::msg::Header msg_header_;

  bool REFINE_MODE_;

  double MAX_RANGE_;
  double MIN_RANGE_;
  double TGF_RESOLUTION_;

  int NUM_ITER_;
  int NUM_LRP_;
  int NUM_MIN_POINTS_;

  double TH_SEEDS_;
  double TH_DIST_;
  double TH_OUTLIER_;

  double TH_NORMAL_;
  double TH_WEIGHT_;
  double TH_LCC_NORMAL_SIMILARITY_;
  double TH_LCC_PLANAR_MODEL_DIST_;
  double TH_OBSTACLE_HEIGHT_;

  TriGridField trigrid_field_;
  std::vector<TriGridEdge> trigrid_edges_;
  std::vector<std::vector<TriGridCorner>> trigrid_corners_;
  std::vector<std::vector<TriGridCorner>> trigrid_centers_;

  pcl::PointCloud<PointType> empty_cloud_;
  TriGridNode empty_trigrid_node_;
  GridNode empty_grid_nodes_;
  TriGridCorner empty_trigrid_corner_;
  TriGridCorner empty_trigrid_center_;

  pcl::PointCloud<PointType> ptCloud_tgfwise_ground_;
  pcl::PointCloud<PointType> ptCloud_tgfwise_nonground_;
  pcl::PointCloud<PointType> ptCloud_tgfwise_outliers_;
  pcl::PointCloud<PointType> ptCloud_tgfwise_obstacle_;
  pcl::PointCloud<PointType> ptCloud_nodewise_ground_;
  pcl::PointCloud<PointType> ptCloud_nodewise_nonground_;
  pcl::PointCloud<PointType> ptCloud_nodewise_outliers_;
  pcl::PointCloud<PointType> ptCloud_nodewise_obstacle_;

public:
  void estimateGround(const pcl::PointCloud<PointType> &cloud_in,
                      pcl::PointCloud<PointType> &cloudGround_out,
                      pcl::PointCloud<PointType> &cloudNonground_out,
                      double &time_taken);

  TriGridIdx getTriGridIdx(const float &x_in, const float &y_in) {

    int r_i = static_cast<int>((x_in - tgf_min_x) / TGF_RESOLUTION_);
    int c_i = static_cast<int>((y_in - tgf_min_y) / TGF_RESOLUTION_);
    int t_i = 0;

    double angle =
        atan2(y_in - (c_i * TGF_RESOLUTION_ + TGF_RESOLUTION_ / 2 + tgf_min_y),
              x_in - (r_i * TGF_RESOLUTION_ + TGF_RESOLUTION_ / 2 + tgf_min_x));

    if (angle >= (M_PI / 4) && angle < (3 * M_PI / 4)) {
      t_i = 1;
    } else if (angle >= (-M_PI / 4) && angle < (M_PI / 4)) {
      t_i = 0;
    } else if (angle >= (-3 * M_PI / 4) && angle < (-M_PI / 4)) {
      t_i = 3;
    } else {
      t_i = 2;
    }

    return TriGridIdx{r_i, c_i, t_i};
  };

  TriGridNode getTriGridNode(const float &x_in, const float &y_in) {
    TriGridNode node;
    TriGridIdx node_idx = getTriGridIdx(x_in, y_in);
    node = trigrid_field_[node_idx.row][node_idx.col][node_idx.tri];
    return node;
  };

  TriGridNode getTriGridNode(const TriGridIdx &tgf_idx) {
    TriGridNode node;
    node = trigrid_field_[tgf_idx.row][tgf_idx.col][tgf_idx.tri];
    return node;
  };

  bool is_traversable(const float &x_in, const float &y_in) {
    TriGridNode node = getTriGridNode(x_in, y_in);
    if (node.node_type == GROUND) {
      return true;
    } else {
      return false;
    }
  };

  pcl::PointCloud<PointType> getObstaclePC() {
    pcl::PointCloud<PointType> cloud_obstacle;
    cloud_obstacle = ptCloud_tgfwise_obstacle_;
    return cloud_obstacle;
  };

private:
  double tgf_max_x, tgf_max_y, tgf_min_x, tgf_min_y;
  double rows_, cols_;

  void initTriGridField(TriGridField &tgf_in);

  void initTriGridCorners(
      std::vector<std::vector<TriGridCorner>> &trigrid_corners_in,
      std::vector<std::vector<TriGridCorner>> &trigrid_centers_in);

  double xy_2Dradius(double x, double y) { return sqrt(x * x + y * y); };

  bool filterPoint(const PointType &pt_in) {
    double xy_range = xy_2Dradius(pt_in.x, pt_in.y);
    return (xy_range >= MAX_RANGE_ || xy_range <= MIN_RANGE_) ? true : false;
  };

  void clearTriGridField(TriGridField &tgf_in);

  void clearTriGridCorners(
      std::vector<std::vector<TriGridCorner>> &trigrid_corners_in,
      std::vector<std::vector<TriGridCorner>> &trigrid_centers_in);

  void embedCloudToTriGridField(const pcl::PointCloud<PointType> &cloud_in,
                                TriGridField &tgf_out);

  void extractInitialSeeds(const pcl::PointCloud<PointType> &p_sorted,
                           pcl::PointCloud<PointType> &init_seeds);

  void estimatePlanarModel(const pcl::PointCloud<PointType> &ground_in,
                           TriGridNode &node_out);

  void modelPCAbasedTerrain(TriGridNode &node_in);

  double calcNodeWeight(const TriGridNode &node_in) {
    double weight = 0;

    weight = (node_in.singular_values[0] + node_in.singular_values[1]) *
             node_in.singular_values[1] /
             (node_in.singular_values[0] * node_in.singular_values[2] + 0.001);

    return weight;
  };

  void modelNodeWiseTerrain(TriGridField &tgf_in);

  void findDominantNode(const TriGridField &tgf_in, TriGridIdx &node_idx_out);

  void searchNeighborNodes(const TriGridIdx &cur_idx,
                           std::vector<TriGridIdx> &neighbor_idxs);

  bool LocalConvecityConcavity(const TriGridField &tgf,
                               const TriGridIdx &cur_node_idx,
                               const TriGridIdx &neighbor_idx,
                               double &thr_local_normal,
                               double &thr_local_dist);

  void BreadthFirstTraversableGraphSearch(TriGridField &tgf_in);

  double getCornerWeight(const TriGridNode &node_in,
                         const pcl::PointXYZ &tgt_corner) {
    double xy_dist = sqrt((node_in.mean_pt[0] - tgt_corner.x) *
                              (node_in.mean_pt[0] - tgt_corner.x) +
                          (node_in.mean_pt[1] - tgt_corner.y) *
                              (node_in.mean_pt[1] - tgt_corner.y));
    return (node_in.weight / xy_dist);
  };

  void setTGFCornersCenters(
      const TriGridField &tgf_in,
      std::vector<std::vector<TriGridCorner>> &trigrid_corners_out,
      std::vector<std::vector<TriGridCorner>> &trigrid_centers_out);

  TriGridCorner getMeanCorner(const TriGridCorner &corners_in);

  void updateTGFCornersCenters(
      std::vector<std::vector<TriGridCorner>> &trigrid_corners_out,
      std::vector<std::vector<TriGridCorner>> &trigrid_centers_out);

  Eigen::Vector3f convertCornerToEigen(TriGridCorner &corner_in) {
    Eigen::Vector3f corner_out;
    if (corner_in.zs.size() != corner_in.weights.size()) {
      std::cout << "ERROR in corners" << std::endl;
    }
    corner_out[0] = corner_in.x;
    corner_out[1] = corner_in.y;
    corner_out[2] = corner_in.zs[0];
    return corner_out;
  };

  void revertTraversableNodes(
      std::vector<std::vector<TriGridCorner>> &trigrid_corners_in,
      std::vector<std::vector<TriGridCorner>> &trigrid_centers_in,
      TriGridField &tgf_out);

  void fitTGFWiseTraversableTerrainModel(
      TriGridField &tgf,
      std::vector<std::vector<TriGridCorner>> &trigrid_corners,
      std::vector<std::vector<TriGridCorner>> &trigrid_centers);

  void segmentNodeGround(const TriGridNode &node_in,
                         pcl::PointCloud<PointType> &node_ground_out,
                         pcl::PointCloud<PointType> &node_nonground_out,
                         pcl::PointCloud<PointType> &node_obstacle_out,
                         pcl::PointCloud<PointType> &node_outlier_out);

  void segmentTGFGround(const TriGridField &tgf_in,
                        pcl::PointCloud<PointType> &ground_cloud_out,
                        pcl::PointCloud<PointType> &nonground_cloud_out,
                        pcl::PointCloud<PointType> &obstacle_cloud_out,
                        pcl::PointCloud<PointType> &outlier_cloud_out);

  // ROS2 INTERFACE
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

  void show_parameters() {
    RCLCPP_INFO(this->get_logger(), "max_range: %f", MAX_RANGE_);
    RCLCPP_INFO(this->get_logger(), "min_range: %f", MIN_RANGE_);
    RCLCPP_INFO(this->get_logger(), "resolution: %f", TGF_RESOLUTION_);
    RCLCPP_INFO(this->get_logger(), "num_iter: %d", NUM_ITER_);
    RCLCPP_INFO(this->get_logger(), "num_lpr: %d", NUM_LRP_);
    RCLCPP_INFO(this->get_logger(), "num_min_pts: %d", NUM_MIN_POINTS_);
    RCLCPP_INFO(this->get_logger(), "th_seeds: %f", TH_SEEDS_);
    RCLCPP_INFO(this->get_logger(), "th_dist: %f", TH_DIST_);
    RCLCPP_INFO(this->get_logger(), "th_outlier: %f", TH_OUTLIER_);
    RCLCPP_INFO(this->get_logger(), "th_normal: %f", TH_NORMAL_);
    RCLCPP_INFO(this->get_logger(), "th_weight: %f", TH_WEIGHT_);
    RCLCPP_INFO(this->get_logger(), "th_lcc_normal_similiarity: %f",
                    TH_LCC_NORMAL_SIMILARITY_);
    RCLCPP_INFO(this->get_logger(), "th_lcc_planar_model_dist: %f",
                TH_LCC_PLANAR_MODEL_DIST_);
    RCLCPP_INFO(this->get_logger(), "th_obstacle: %f", TH_OBSTACLE_HEIGHT_);
    RCLCPP_INFO(this->get_logger(), "refine_mode: %d", REFINE_MODE_);
  };

public:
  explicit TravelGroundFilterComponent(const rclcpp::NodeOptions & options);;
};
} // namespace ground_segmentation

#endif // TRAVEL_GROUND_FILTER_NODELET_HPP
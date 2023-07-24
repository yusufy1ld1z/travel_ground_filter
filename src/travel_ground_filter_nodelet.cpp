#include "ground_segmentation/travel_ground_filter_nodelet.hpp"

#include <iostream>
#include <math.h>
#include <queue>
#include <rclcpp/node_options.hpp>
#include <vector>

namespace ground_segmentation {

  TravelGroundFilterComponent::TravelGroundFilterComponent(const rclcpp::NodeOptions& options) 
      : Node("TravelGroundFilter", options){
      using std::placeholders::_1;
      using std::chrono_literals::operator""ms;

      pc_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
          "cloud", rclcpp::SensorDataQoS());
      pc_ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
          "ground", rclcpp::SensorDataQoS());
      pc_nonground_publisher_ =
          this->create_publisher<sensor_msgs::msg::PointCloud2>(
              "nonground", rclcpp::SensorDataQoS());

      pc_cloud_subscriber_ =
          this->create_subscription<sensor_msgs::msg::PointCloud2>(
              "/sensing/lidar/concatenated/pointcloud", rclcpp::SensorDataQoS(),
              std::bind(&TravelGroundFilterComponent::point_cloud_callback, this, _1));

      MAX_RANGE_ = this->declare_parameter("max_range", 10.0);
      MIN_RANGE_ = this->declare_parameter("min_range", 0.5);
      TGF_RESOLUTION_ = this->declare_parameter("resolution", 2.0);
      NUM_ITER_ = this->declare_parameter("num_iter", 3);
      NUM_LRP_ = this->declare_parameter("num_lpr", 3);
      NUM_MIN_POINTS_ = this->declare_parameter("num_min_pts", 3);
      TH_SEEDS_ = this->declare_parameter("th_seeds", 1.0);
      TH_DIST_ = this->declare_parameter("th_dist", 0.1);
      TH_OUTLIER_ = this->declare_parameter("th_outlier", 0.1);
      TH_NORMAL_ = this->declare_parameter("th_normal", 0.1);
      TH_WEIGHT_ = this->declare_parameter("th_weight", 0.1);
      TH_LCC_NORMAL_SIMILARITY_ = this->declare_parameter("th_lcc_normal", 0.1);
      TH_LCC_PLANAR_MODEL_DIST_ = this->declare_parameter("th_lcc_planar", 0.1);
      TH_OBSTACLE_HEIGHT_ = this->declare_parameter("th_obstacle", 0.1);
      REFINE_MODE_ = this->declare_parameter("refine_mode", false);
      std::cout << "" << std::endl;

      initTriGridField(trigrid_field_);
      initTriGridCorners(trigrid_corners_, trigrid_centers_);

      ptCloud_tgfwise_ground_.clear();
      ptCloud_tgfwise_ground_.reserve(PTCLOUD_SIZE);
      ptCloud_tgfwise_nonground_.clear();
      ptCloud_tgfwise_nonground_.reserve(PTCLOUD_SIZE);
      ptCloud_tgfwise_outliers_.clear();
      ptCloud_tgfwise_outliers_.reserve(PTCLOUD_SIZE);
      ptCloud_tgfwise_obstacle_.clear();
      ptCloud_tgfwise_obstacle_.reserve(PTCLOUD_SIZE);

      ptCloud_nodewise_ground_.clear();
      ptCloud_nodewise_ground_.reserve(NODEWISE_PTCLOUDSIZE);
      ptCloud_nodewise_nonground_.clear();
      ptCloud_nodewise_nonground_.reserve(NODEWISE_PTCLOUDSIZE);
      ptCloud_nodewise_outliers_.clear();
      ptCloud_nodewise_outliers_.reserve(NODEWISE_PTCLOUDSIZE);
      ptCloud_nodewise_obstacle_.clear();
      ptCloud_nodewise_obstacle_.reserve(NODEWISE_PTCLOUDSIZE);

      show_parameters();
  }

  void TravelGroundFilterComponent::point_cloud_callback(
      const sensor_msgs::msg::PointCloud2::SharedPtr point_cloud) {

    RCLCPP_INFO(this->get_logger(), "Received point cloud");

    pcl::PointCloud<PointType>::Ptr input_cloud(new pcl::PointCloud<PointType>);

    pcl::fromROSMsg(*point_cloud, *input_cloud);

    pcl::PointCloud<PointType>::Ptr ground_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr nonground_cloud(
        new pcl::PointCloud<PointType>);

    double elapsed_time = 0.0;
    this->estimateGround(*input_cloud, *ground_cloud,
                                                *nonground_cloud, elapsed_time);
    RCLCPP_INFO(this->get_logger(), "Total time: %f", elapsed_time);

    sensor_msgs::msg::PointCloud2 ground_cloud_msg;
    pcl::toROSMsg(*ground_cloud, ground_cloud_msg);
    ground_cloud_msg.header.frame_id = "base_link";
    ground_cloud_msg.header.stamp = point_cloud->header.stamp;

    pc_ground_publisher_->publish(ground_cloud_msg);

    sensor_msgs::msg::PointCloud2 nonground_cloud_msg;
    pcl::toROSMsg(*nonground_cloud, nonground_cloud_msg);
    nonground_cloud_msg.header.frame_id = "base_link";
    nonground_cloud_msg.header.stamp = point_cloud->header.stamp;
    pc_nonground_publisher_->publish(nonground_cloud_msg);
  }

void TravelGroundFilterComponent::estimateGround(
    const pcl::PointCloud<PointType> &cloud_in,
    pcl::PointCloud<PointType> &cloudGround_out,
    pcl::PointCloud<PointType> &cloudNonground_out, double &time_taken) {

  // 0. Init
  static time_t start, end;
  cloud_header_ = cloud_in.header;
  pcl_conversions::fromPCL(cloud_header_, msg_header_);
  start = clock();
  ptCloud_tgfwise_outliers_.clear();
  ptCloud_tgfwise_outliers_.reserve(cloud_in.size());

  // 1. Embed PointCloud to TriGridField
  clearTriGridField(trigrid_field_);
  clearTriGridCorners(trigrid_corners_, trigrid_centers_);

  embedCloudToTriGridField(cloud_in, trigrid_field_);

  // 2. Node-wise Terrain Modeling
  modelNodeWiseTerrain(trigrid_field_);

  // 3. Breadth-first Traversable Graph Search
  BreadthFirstTraversableGraphSearch(trigrid_field_);
  setTGFCornersCenters(trigrid_field_, trigrid_corners_, trigrid_centers_);

  // 4. TGF-wise Traversable Terrain Model Fitting
  if (REFINE_MODE_) {
    fitTGFWiseTraversableTerrainModel(trigrid_field_, trigrid_corners_,
                                      trigrid_centers_);
  }

  // 5. Ground Segmentation
  segmentTGFGround(trigrid_field_, ptCloud_tgfwise_ground_,
                   ptCloud_tgfwise_nonground_, ptCloud_tgfwise_obstacle_,
                   ptCloud_tgfwise_outliers_);
  cloudGround_out = ptCloud_tgfwise_ground_;
  cloudNonground_out = ptCloud_tgfwise_nonground_;
  cloudGround_out.header = cloudNonground_out.header = cloud_header_;

  end = clock();
  time_taken = (double)(end - start) / CLOCKS_PER_SEC;

  // 6. Publish Results and Visualization

  return;
}

void TravelGroundFilterComponent::initTriGridField(TriGridField &tgf_in) {

  tgf_max_x = MAX_RANGE_;
  tgf_max_y = MAX_RANGE_;

  tgf_min_x = -MAX_RANGE_;
  tgf_min_y = -MAX_RANGE_;

  rows_ = static_cast<int>((tgf_max_x - tgf_min_x) / TGF_RESOLUTION_);
  cols_ = static_cast<int>((tgf_max_y - tgf_min_y) / TGF_RESOLUTION_);

  // std::cout << "ROWS and COLS " << rows_ << "\t" << cols_ << std::endl;

  empty_cloud_.clear();
  empty_cloud_.reserve(PTCLOUD_SIZE);

  // Set Node structure
  empty_trigrid_node_.node_type = UNKNOWN;
  empty_trigrid_node_.ptCloud.clear();
  empty_trigrid_node_.ptCloud.reserve(NODEWISE_PTCLOUDSIZE);

  empty_trigrid_node_.is_curr_data = false;
  empty_trigrid_node_.need_recheck = false;
  empty_trigrid_node_.is_visited = false;
  empty_trigrid_node_.is_rejection = false;

  empty_trigrid_node_.check_life = 10;
  empty_trigrid_node_.depth = -1;

  empty_trigrid_node_.normal;
  empty_trigrid_node_.mean_pt;
  empty_trigrid_node_.d = 0;

  empty_trigrid_node_.singular_values;
  empty_trigrid_node_.eigen_vectors;
  empty_trigrid_node_.weight = 0;

  empty_trigrid_node_.th_dist_d = 0;
  empty_trigrid_node_.th_outlier_d = 0;
  // Set TriGridField
  tgf_in.clear();
  std::vector<GridNode> vec_gridnode;

  for (int i = 0; i < 4; i++)
    empty_grid_nodes_.emplace_back(empty_trigrid_node_);

  for (int j = 0; j < cols_; j++) {
    vec_gridnode.emplace_back(empty_grid_nodes_);
  }
  for (int k = 0; k < rows_; k++) {
    tgf_in.emplace_back(vec_gridnode);
  }
  return;
}

void TravelGroundFilterComponent::initTriGridCorners(
    std::vector<std::vector<TriGridCorner>> &trigrid_corners_in,
    std::vector<std::vector<TriGridCorner>> &trigrid_centers_in) {

  // Set TriGridCorner
  empty_trigrid_corner_.x = empty_trigrid_corner_.y = 0.0;
  empty_trigrid_corner_.zs.clear();
  empty_trigrid_corner_.zs.reserve(8);
  empty_trigrid_corner_.weights.clear();
  empty_trigrid_corner_.weights.reserve(8);

  empty_trigrid_center_.x = empty_trigrid_center_.y = 0.0;
  empty_trigrid_center_.zs.clear();
  empty_trigrid_center_.zs.reserve(4);
  empty_trigrid_center_.weights.clear();
  empty_trigrid_center_.weights.reserve(4);

  trigrid_corners_in.clear();
  trigrid_centers_in.clear();

  std::vector<TriGridCorner> col_corners(cols_ + 1, empty_trigrid_corner_);
  std::vector<TriGridCorner> col_centers(cols_, empty_trigrid_center_);

  trigrid_corners_in.assign(rows_ + 1, col_corners);
  trigrid_centers_in.assign(rows_, col_centers);
  return;
}

void TravelGroundFilterComponent::clearTriGridField(TriGridField &tgf_in) {

  for (int r_i = 0; r_i < rows_; r_i++) {
    std::fill(tgf_in[r_i].begin(), tgf_in[r_i].end(), empty_grid_nodes_);
  }
  return;
}

void TravelGroundFilterComponent::clearTriGridCorners(
    std::vector<std::vector<TriGridCorner>> &trigrid_corners_in,
    std::vector<std::vector<TriGridCorner>> &trigrid_centers_in) {

  TriGridCorner tmp_corner = empty_trigrid_corner_;
  TriGridCorner tmp_center = empty_trigrid_center_;
  for (int r_i = 0; r_i < rows_ + 1; r_i++) {
    for (int c_i = 0; c_i < cols_ + 1; c_i++) {
      tmp_corner.x = (r_i)*TGF_RESOLUTION_ + tgf_min_x;
      tmp_corner.y = (c_i)*TGF_RESOLUTION_ + tgf_min_y;
      tmp_corner.zs.clear();
      tmp_corner.weights.clear();
      trigrid_corners_in[r_i][c_i] = tmp_corner;
      if (r_i < rows_ && c_i < cols_) {
        tmp_center.x = (r_i + 0.5) * TGF_RESOLUTION_ + tgf_min_x;
        tmp_center.y = (c_i + 0.5) * TGF_RESOLUTION_ + tgf_min_y;
        tmp_center.zs.clear();
        tmp_center.weights.clear();
        trigrid_centers_in[r_i][c_i] = tmp_center;
      }
    }
  }
  return;
}

void TravelGroundFilterComponent::embedCloudToTriGridField(
    const pcl::PointCloud<PointType> &cloud_in, TriGridField &tgf_out) {

  for (const auto &pt : cloud_in.points) {
    if (filterPoint(pt)) {
      ptCloud_tgfwise_outliers_.points.push_back(pt);
      continue;
    }
    int r_i = static_cast<int>((pt.x - tgf_min_x) / TGF_RESOLUTION_);
    int c_i = static_cast<int>((pt.y - tgf_min_y) / TGF_RESOLUTION_);

    if (r_i >= 0 && r_i < rows_ && c_i >= 0 && c_i < cols_) {
      double angle = atan2(
          pt.y - (c_i * TGF_RESOLUTION_ + TGF_RESOLUTION_ / 2 + tgf_min_y),
          pt.x - (r_i * TGF_RESOLUTION_ + TGF_RESOLUTION_ / 2 + tgf_min_x));

      int t_i = 0;
      if (angle >= (M_PI / 4) && angle < (3 * M_PI / 4)) {
        t_i = 1; // left side
      } else if (angle >= (-M_PI / 4) && angle < (M_PI / 4)) {
        t_i = 0; // upper side
      } else if (angle >= (-3 * M_PI / 4) && angle < (-M_PI / 4)) {
        t_i = 3; // right side
      } else {
        t_i = 2; // lower side
      }

      tgf_out[r_i][c_i][t_i].ptCloud.emplace_back(pt);

      if (!tgf_out[r_i][c_i][t_i].is_curr_data) {
        tgf_out[r_i][c_i][t_i].is_curr_data = true;
      }
    } else {
      ptCloud_tgfwise_outliers_.points.push_back(pt);
    }
  }
  return;
}

void TravelGroundFilterComponent::extractInitialSeeds(
    const pcl::PointCloud<PointType> &p_sorted,
    pcl::PointCloud<PointType> &init_seeds) {

  // Function for uniform mode
  init_seeds.points.clear();

  // LPR is the mean of Low Point Representative
  double sum = 0;
  int cnt = 0;

  // Calculate the mean height value and find the upper and lower bounds for
  // seeds
  double upper_bound = 0, lower_bound = 0;
  for (const PointType &point : p_sorted.points) {
    sum += point.z;
    cnt++;
    if (cnt == NUM_LRP_) {
      lower_bound = sum / cnt - TH_OUTLIER_;
      upper_bound = sum / cnt + TH_SEEDS_;
      break;
    }
  }

  // Extract initial seeds within the bounds
  for (const PointType &point : p_sorted.points) {
    double z = point.z;
    if (z >= lower_bound && z <= upper_bound) {
      init_seeds.points.push_back(point);
    }
  }
  return;
}

void TravelGroundFilterComponent::estimatePlanarModel(
    const pcl::PointCloud<PointType> &ground_in, TriGridNode &node_out) {

  // function for uniform mode
  Eigen::Matrix3f cov_;
  Eigen::Vector4f pc_mean_;
  pcl::computeMeanAndCovarianceMatrix(ground_in, cov_, pc_mean_);

  // Singular Value Decomposition: SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(
      cov_, Eigen::DecompositionOptions::ComputeFullU);

  // Use the least singular vector as normal
  node_out.eigen_vectors = svd.matrixU();
  if (node_out.eigen_vectors.col(2)(2, 0) < 0) {
    node_out.eigen_vectors.col(0) *= -1;
    node_out.eigen_vectors.col(2) *= -1;
  }

  node_out.normal = node_out.eigen_vectors.col(2);
  node_out.singular_values = svd.singularValues();

  // mean ground seeds value
  node_out.mean_pt = pc_mean_.head<3>();
  // according to normal.T*[x,y,z] = -d
  node_out.d = -(node_out.normal.transpose() * node_out.mean_pt)(0, 0);

  // set distance theshold to 'th_dist - d'
  node_out.th_dist_d = TH_DIST_ - node_out.d;
  node_out.th_outlier_d = -node_out.d - TH_OUTLIER_;

  return;
}

void TravelGroundFilterComponent::modelPCAbasedTerrain(TriGridNode &node_in) {
  // Initailization
  if (!ptCloud_nodewise_ground_.empty())
    ptCloud_nodewise_ground_.clear();

  // Tri Grid Initialization
  // When to initialize the planar model, we don't have prior. so outlier is
  // removed in heuristic parameter.
  pcl::PointCloud<PointType> sort_ptCloud = node_in.ptCloud;

  // sort in z-coordinate
  sort(sort_ptCloud.points.begin(), sort_ptCloud.end(), [](const PointType& a, const PointType& b){ return a.z < b.z; });

  // Set init seeds
  extractInitialSeeds(sort_ptCloud, ptCloud_nodewise_ground_);

  Eigen::MatrixXf points(sort_ptCloud.points.size(), 3);
  int j = 0;
  for (auto &p : sort_ptCloud.points) {
    points.row(j++) << p.x, p.y, p.z;
  }
  // Extract Ground
  for (int i = 0; i < NUM_ITER_; i++) {
    estimatePlanarModel(ptCloud_nodewise_ground_, node_in);
    if (ptCloud_nodewise_ground_.size() < 3) {

      node_in.node_type = NONGROUND;
      break;
    }
    ptCloud_nodewise_ground_.clear();
    // threshold filter
    Eigen::VectorXf result = points * node_in.normal;
    for (int r = 0; r < result.rows(); r++) {
      if (i < NUM_ITER_ - 1) {
        if (result[r] < node_in.th_dist_d) {
          ptCloud_nodewise_ground_.push_back(sort_ptCloud.points[r]);
        }
      } else {
        // Final interation
        node_in.node_type =
            (node_in.normal(2, 0) < TH_NORMAL_) ? NONGROUND : GROUND;
      }
    }
  }
  return;
}

void TravelGroundFilterComponent::modelNodeWiseTerrain(TriGridField &tgf_in) {

  for (int r_i = 0; r_i < rows_; r_i++) {
    for (int c_i = 0; c_i < cols_; c_i++) {
      for (int s_i = 0; s_i < 4; s_i++) {
        if (tgf_in[r_i][c_i][s_i].is_curr_data) {
          if (tgf_in[r_i][c_i][s_i].ptCloud.size() < NUM_MIN_POINTS_) {
            tgf_in[r_i][c_i][s_i].node_type = UNKNOWN;
            continue;
          } else {
            modelPCAbasedTerrain(tgf_in[r_i][c_i][s_i]);
            if (tgf_in[r_i][c_i][s_i].node_type == GROUND) {
              tgf_in[r_i][c_i][s_i].weight =
                  calcNodeWeight(tgf_in[r_i][c_i][s_i]);
            }
          }
        }
      }
    }
  }

  return;
}

void TravelGroundFilterComponent::findDominantNode(const TriGridField &tgf_in,
                                                   TriGridIdx &node_idx_out) {
  // Find the dominant node
  TriGridIdx max_tri_idx;
  TriGridIdx ego_idx;
  ego_idx.row = (int)((0 - tgf_min_x) / TGF_RESOLUTION_);
  ego_idx.col = (int)((0 - tgf_min_y) / TGF_RESOLUTION_);
  ego_idx.tri = 0;

  max_tri_idx = ego_idx;
  for (int r_i = ego_idx.row - 2; r_i < ego_idx.row + 2; r_i++) {
    for (int c_i = ego_idx.col - 2; c_i < ego_idx.col + 2; c_i++) {
      for (int s_i = 0; s_i < 4; s_i++) {
        if (tgf_in[r_i][c_i][s_i].is_curr_data) {
          if (tgf_in[r_i][c_i][s_i].node_type == GROUND) {
            if (tgf_in[r_i][c_i][s_i].weight >
                tgf_in[max_tri_idx.row][max_tri_idx.row][max_tri_idx.tri]
                    .weight) {
              max_tri_idx.row = r_i;
              max_tri_idx.col = c_i;
              max_tri_idx.tri = s_i;
            }
          }
        }
      }
    }
  }
  node_idx_out = max_tri_idx;
  return;
}

void TravelGroundFilterComponent::searchNeighborNodes(
    const TriGridIdx &cur_idx, std::vector<TriGridIdx> &neighbor_idxs) {
  neighbor_idxs.clear();
  neighbor_idxs.reserve(14);
  int r_i = cur_idx.row;
  int c_i = cur_idx.col;
  int s_i = cur_idx.tri;

  std::vector<TriGridIdx> tmp_neighbors;
  tmp_neighbors.clear();
  tmp_neighbors.reserve(14);

  TriGridIdx neighbor_idx;
  for (int s_i = 0; s_i < 4; s_i++) {
    if (s_i == cur_idx.tri)
      continue;

    neighbor_idx = cur_idx;
    neighbor_idx.tri = s_i;
    tmp_neighbors.push_back(neighbor_idx);
  }

  switch (s_i) {
  case 0:
    tmp_neighbors.push_back({r_i + 1, c_i + 1, 2});
    tmp_neighbors.push_back({r_i + 1, c_i + 1, 3});
    tmp_neighbors.push_back({r_i + 1, c_i, 1});
    tmp_neighbors.push_back({r_i + 1, c_i, 2});
    tmp_neighbors.push_back({r_i + 1, c_i, 3});
    tmp_neighbors.push_back({r_i + 1, c_i - 1, 1});
    tmp_neighbors.push_back({r_i + 1, c_i - 1, 2});
    tmp_neighbors.push_back({r_i, c_i + 1, 0});
    tmp_neighbors.push_back({r_i, c_i + 1, 3});
    tmp_neighbors.push_back({r_i, c_i - 1, 0});
    tmp_neighbors.push_back({r_i, c_i - 1, 1});
    break;
  case 1:
    tmp_neighbors.push_back({r_i + 1, c_i + 1, 2});
    tmp_neighbors.push_back({r_i + 1, c_i + 1, 3});
    tmp_neighbors.push_back({r_i + 1, c_i, 1});
    tmp_neighbors.push_back({r_i + 1, c_i, 2});
    tmp_neighbors.push_back({r_i, c_i + 1, 0});
    tmp_neighbors.push_back({r_i, c_i + 1, 2});
    tmp_neighbors.push_back({r_i, c_i + 1, 3});
    tmp_neighbors.push_back({r_i - 1, c_i + 1, 0});
    tmp_neighbors.push_back({r_i - 1, c_i + 1, 3});
    tmp_neighbors.push_back({r_i - 1, c_i + 1, 0});
    tmp_neighbors.push_back({r_i - 1, c_i, 1});
    break;
  case 2:
    tmp_neighbors.push_back({r_i, c_i + 1, 2});
    tmp_neighbors.push_back({r_i, c_i + 1, 3});
    tmp_neighbors.push_back({r_i, c_i - 1, 1});
    tmp_neighbors.push_back({r_i, c_i - 1, 2});
    tmp_neighbors.push_back({r_i - 1, c_i + 1, 0});
    tmp_neighbors.push_back({r_i - 1, c_i + 1, 3});
    tmp_neighbors.push_back({r_i - 1, c_i, 0});
    tmp_neighbors.push_back({r_i - 1, c_i, 1});
    tmp_neighbors.push_back({r_i - 1, c_i, 3});
    tmp_neighbors.push_back({r_i - 1, c_i - 1, 0});
    tmp_neighbors.push_back({r_i - 1, c_i - 1, 1});
    break;
  case 3:
    tmp_neighbors.push_back({r_i + 1, c_i, 2});
    tmp_neighbors.push_back({r_i + 1, c_i, 3});
    tmp_neighbors.push_back({r_i + 1, c_i - 1, 1});
    tmp_neighbors.push_back({r_i + 1, c_i - 1, 2});
    tmp_neighbors.push_back({r_i, c_i - 1, 0});
    tmp_neighbors.push_back({r_i, c_i - 1, 1});
    tmp_neighbors.push_back({r_i, c_i - 1, 2});
    tmp_neighbors.push_back({r_i - 1, c_i, 0});
    tmp_neighbors.push_back({r_i - 1, c_i, 3});
    tmp_neighbors.push_back({r_i - 1, c_i - 1, 0});
    tmp_neighbors.push_back({r_i - 1, c_i - 1, 1});
    break;
  default:
    break;
  }

  for (int n_i = 0; n_i < static_cast<int>(tmp_neighbors.size()); n_i++) {
    if (tmp_neighbors[n_i].row >= 0 && tmp_neighbors[n_i].row < rows_ &&
        tmp_neighbors[n_i].col >= 0 && tmp_neighbors[n_i].col < cols_)
      neighbor_idxs.push_back(tmp_neighbors[n_i]);
  }
}

bool TravelGroundFilterComponent::LocalConvecityConcavity(
    const TriGridField &tgf, const TriGridIdx &cur_node_idx,
    const TriGridIdx &neighbor_idx, double &thr_local_normal,
    double &thr_local_dist) {

  const TriGridNode current_node =
      tgf[cur_node_idx.row][cur_node_idx.col][cur_node_idx.tri];
  const TriGridNode neighbor_node =
      tgf[neighbor_idx.row][neighbor_idx.col][neighbor_idx.tri];

  Eigen::Vector3f normal_src = current_node.normal;
  Eigen::Vector3f normal_tgt = neighbor_node.normal;
  Eigen::Vector3f meanPt_diff_s2t =
      neighbor_node.mean_pt - current_node.mean_pt;

  double diff_norm = meanPt_diff_s2t.norm();
  double dist_s2t = normal_src.dot(meanPt_diff_s2t);
  double dist_t2s = normal_tgt.dot(-meanPt_diff_s2t);

  double normal_similarity = normal_src.dot(normal_tgt);
  double TH_NORMAL_cos_similarity = sin(diff_norm * thr_local_normal);
  double TH_DIST_to_planar = diff_norm * sin(thr_local_dist);

  return (normal_similarity >= (1 - TH_NORMAL_cos_similarity)) &&
         (abs(dist_s2t) <= TH_DIST_to_planar) &&
         (abs(dist_t2s) <= TH_DIST_to_planar);
}

void TravelGroundFilterComponent::BreadthFirstTraversableGraphSearch(
    TriGridField &tgf_in) {

  // Find the dominant node
  std::queue<TriGridIdx> searching_idx_queue;
  TriGridIdx dominant_node_idx;
  findDominantNode(tgf_in, dominant_node_idx);
  tgf_in[dominant_node_idx.row][dominant_node_idx.col][dominant_node_idx.tri]
      .is_visited = true;
  tgf_in[dominant_node_idx.row][dominant_node_idx.col][dominant_node_idx.tri]
      .depth = 0;
  tgf_in[dominant_node_idx.row][dominant_node_idx.col][dominant_node_idx.tri]
      .node_type = GROUND;

  searching_idx_queue.push(dominant_node_idx);

  double max_planar_height = 0;
  trigrid_edges_.clear();
  trigrid_edges_.reserve(rows_ * cols_ * 4);
  TriGridEdge cur_edge;
  TriGridIdx current_node_idx;
  while (!searching_idx_queue.empty()) {
    // set current node
    current_node_idx = searching_idx_queue.front();
    searching_idx_queue.pop();

    // search the neighbor nodes
    std::vector<TriGridIdx> neighbor_idxs;
    searchNeighborNodes(current_node_idx, neighbor_idxs);

    // set the traversable edges
    for (const auto &n_i : neighbor_idxs) {
      // if the neighbor node is traversable, add it to the queue

      auto &neighbor_node = tgf_in[n_i.row][n_i.col][n_i.tri];

      if (neighbor_node.depth >= 0 || neighbor_node.is_visited ||
          neighbor_node.node_type != GROUND) {
        continue;
      }

      neighbor_node.is_visited = true;

      if (!LocalConvecityConcavity(tgf_in, current_node_idx, n_i,
                                   TH_LCC_NORMAL_SIMILARITY_,
                                   TH_LCC_PLANAR_MODEL_DIST_)) {
        neighbor_node.is_rejection = true;
        neighbor_node.node_type = NONGROUND;

        if (neighbor_node.check_life > 0) {
          neighbor_node.check_life -= 1;
          neighbor_node.need_recheck = true;
        } else {
          neighbor_node.need_recheck = false;
        }
        continue;
      }

      if (max_planar_height < neighbor_node.mean_pt[2])
        max_planar_height = neighbor_node.mean_pt[2];

      neighbor_node.node_type = GROUND;
      neighbor_node.is_rejection = false;
      neighbor_node.depth = tgf_in[current_node_idx.row][current_node_idx.col]
                                  [current_node_idx.tri]
                                      .depth +
                            1;
      searching_idx_queue.push(n_i);
    }

    if (searching_idx_queue.empty()) {
      // set the new dominant node
      for (int r_i = 0; r_i < rows_; r_i++) {
        for (int c_i = 0; c_i < cols_; c_i++) {
          for (int s_i = 0; s_i < static_cast<int>(tgf_in[r_i][c_i].size());
               s_i++) {

            auto &node = tgf_in[r_i][c_i][s_i];
            if (!node.is_visited && node.node_type == GROUND &&
                node.depth < 0) {
              node.depth = 0;
              node.is_visited = true;

              TriGridIdx new_dominant_idx = {r_i, c_i, s_i};
              searching_idx_queue.push(new_dominant_idx);
            }
          }
        }
      }
    }
  }
  return;
}

void TravelGroundFilterComponent::setTGFCornersCenters(
    const TriGridField &tgf_in,
    std::vector<std::vector<TriGridCorner>> &trigrid_corners_out,
    std::vector<std::vector<TriGridCorner>> &trigrid_centers_out) {

  pcl::PointXYZ corner_TL, corner_BL, corner_BR, corner_TR, corner_C;

  for (int r_i = 0; r_i < rows_; r_i++) {
    for (int c_i = 0; c_i < cols_; c_i++) {

      const auto &current_node = tgf_in[r_i][c_i];

      corner_TL.x = trigrid_corners_out[r_i + 1][c_i + 1].x;
      corner_TL.y = trigrid_corners_out[r_i + 1][c_i + 1].y; // LT
      corner_BL.x = trigrid_corners_out[r_i][c_i + 1].x;
      corner_BL.y = trigrid_corners_out[r_i][c_i + 1].y; // LL
      corner_BR.x = trigrid_corners_out[r_i][c_i].x;
      corner_BR.y = trigrid_corners_out[r_i][c_i].y; // RL
      corner_TR.x = trigrid_corners_out[r_i + 1][c_i].x;
      corner_TR.y = trigrid_corners_out[r_i + 1][c_i].y; // RT
      corner_C.x = trigrid_centers_out[r_i][c_i].x;
      corner_C.y = trigrid_centers_out[r_i][c_i].y; // Center

      for (int s_i = 0; s_i < static_cast<int>(current_node.size()); s_i++) {

        const auto &node = tgf_in[r_i][c_i][s_i];
        Eigen::Vector3f normal = node.normal;
        double d = node.d;

        if (node.node_type != GROUND || node.is_rejection || node.depth == -1) {
          continue;
        }
        auto calculateZ = [&normal, &d](const pcl::PointXYZ &corner) {
          return -(normal(0, 0) * corner.x + normal(1, 0) * corner.y + d) /
                 normal(2, 0);
        };

        switch (s_i) {
        case 0: // upper Tri-grid bin
          // RT / LT / C
          trigrid_corners_out[r_i + 1][c_i].zs.push_back(calculateZ(corner_TR));
          trigrid_corners_out[r_i + 1][c_i].weights.push_back(
              getCornerWeight(node, corner_TR));

          trigrid_corners_out[r_i + 1][c_i + 1].zs.push_back(
              calculateZ(corner_TL));
          trigrid_corners_out[r_i + 1][c_i + 1].weights.push_back(
              getCornerWeight(node, corner_TL));

          trigrid_centers_out[r_i][c_i].zs.push_back(calculateZ(corner_C));
          trigrid_centers_out[r_i][c_i].weights.push_back(
              getCornerWeight(node, corner_C));
          break;
        case 1: // left Tri-grid bin
          // LT / LL / C
          trigrid_corners_out[r_i + 1][c_i + 1].zs.push_back(
              calculateZ(corner_TL));
          trigrid_corners_out[r_i + 1][c_i + 1].weights.push_back(
              getCornerWeight(node, corner_TL));

          trigrid_corners_out[r_i][c_i + 1].zs.push_back(calculateZ(corner_BL));
          trigrid_corners_out[r_i][c_i + 1].weights.push_back(
              getCornerWeight(node, corner_BL));

          trigrid_centers_out[r_i][c_i].zs.push_back(calculateZ(corner_C));
          trigrid_centers_out[r_i][c_i].weights.push_back(
              getCornerWeight(node, corner_C));

          break;
        case 2: // lower Tri-grid bin
          // LL / RL / C
          trigrid_corners_out[r_i][c_i + 1].zs.push_back(calculateZ(corner_BL));
          trigrid_corners_out[r_i][c_i + 1].weights.push_back(
              getCornerWeight(node, corner_BL));

          trigrid_corners_out[r_i][c_i].zs.push_back(calculateZ(corner_BR));
          trigrid_corners_out[r_i][c_i].weights.push_back(
              getCornerWeight(node, corner_BR));

          trigrid_centers_out[r_i][c_i].zs.push_back(calculateZ(corner_C));
          trigrid_centers_out[r_i][c_i].weights.push_back(
              getCornerWeight(node, corner_C));

          break;
        case 3: // right Tri-grid bin
          // RL / RT / C
          trigrid_corners_out[r_i][c_i].zs.push_back(calculateZ(corner_BR));
          trigrid_corners_out[r_i][c_i].weights.push_back(
              getCornerWeight(node, corner_BR));

          trigrid_corners_out[r_i + 1][c_i].zs.push_back(calculateZ(corner_TR));
          trigrid_corners_out[r_i + 1][c_i].weights.push_back(
              getCornerWeight(node, corner_TR));

          trigrid_centers_out[r_i][c_i].zs.push_back(calculateZ(corner_C));
          trigrid_centers_out[r_i][c_i].weights.push_back(
              getCornerWeight(node, corner_C));

          break;
        default:
          break;
        }
      }
    }
  }
  return;
}

TriGridCorner
TravelGroundFilterComponent::getMeanCorner(const TriGridCorner &corners_in) {
  // get the mean of the corners

  TriGridCorner corners_out;
  corners_out.x = corners_in.x;
  corners_out.y = corners_in.y;
  corners_out.zs.clear();
  corners_out.weights.clear();

  double weighted_sum_z = 0.0;
  double sum_w = 0.0;
  for (int i = 0; i < static_cast<int>(corners_in.zs.size()); i++) {
    weighted_sum_z += corners_in.zs[i] * corners_in.weights[i];
    sum_w += corners_in.weights[i];
  }

  corners_out.zs.push_back(weighted_sum_z / sum_w);
  corners_out.weights.push_back(sum_w);

  return corners_out;
}

void TravelGroundFilterComponent::updateTGFCornersCenters(
    std::vector<std::vector<TriGridCorner>> &trigrid_corners_out,
    std::vector<std::vector<TriGridCorner>> &trigrid_centers_out) {

  // update corners
  TriGridCorner updated_corner = empty_trigrid_corner_;
  for (int r_i = 0; r_i < rows_ + 1; r_i++) {
    for (int c_i = 0; c_i < cols_ + 1; c_i++) {
      if (!trigrid_corners_out[r_i][c_i].zs.empty() &&
          !trigrid_corners_out[r_i][c_i].weights.empty()) {
        updated_corner = getMeanCorner(trigrid_corners_out[r_i][c_i]);
        trigrid_corners_out[r_i][c_i] = updated_corner;
      } else {
        trigrid_corners_out[r_i][c_i].zs.clear();
        trigrid_corners_out[r_i][c_i].weights.clear();
      }
    }
  }
  // update centers
  TriGridCorner updated_center = empty_trigrid_center_;
  for (int r_i = 0; r_i < rows_; r_i++) {
    for (int c_i = 0; c_i < cols_; c_i++) {
      if (!trigrid_centers_out[r_i][c_i].zs.empty() &&
          !trigrid_centers_out[r_i][c_i].weights.empty()) {
        updated_center = getMeanCorner(trigrid_centers_out[r_i][c_i]);
        trigrid_centers_out[r_i][c_i] = updated_center;
      } else {
        trigrid_centers_out[r_i][c_i].zs.clear();
        trigrid_centers_out[r_i][c_i].weights.clear();
      }
    }
  }

  return;
}

void TravelGroundFilterComponent::revertTraversableNodes(
    std::vector<std::vector<TriGridCorner>> &trigrid_corners_in,
    std::vector<std::vector<TriGridCorner>> &trigrid_centers_in,
    TriGridField &tgf_out) {
  Eigen::Vector3f refined_corner_1, refined_corner_2, refined_center;
  for (int r_i = 0; r_i < rows_; r_i++) {
    for (int c_i = 0; c_i < cols_; c_i++) {
      for (int s_i = 0; s_i < (int)tgf_out[r_i][c_i].size(); s_i++) {
        // set the corners for the current trigrid node
        switch (s_i) {
        case 0:
          if (trigrid_corners_in[r_i + 1][c_i].zs.empty() ||
              trigrid_corners_in[r_i + 1][c_i + 1].zs.empty() ||
              trigrid_centers_in[r_i][c_i].zs.empty()) {
            if (tgf_out[r_i][c_i][s_i].node_type != NONGROUND) {
              tgf_out[r_i][c_i][s_i].node_type = UNKNOWN;
            }
            continue;
          }
          refined_corner_1 =
              convertCornerToEigen(trigrid_corners_in[r_i + 1][c_i]);
          refined_corner_2 =
              convertCornerToEigen(trigrid_corners_in[r_i + 1][c_i + 1]);
          refined_center = convertCornerToEigen(trigrid_centers_in[r_i][c_i]);
          break;
        case 1:
          if (trigrid_corners_in[r_i + 1][c_i + 1].zs.empty() ||
              trigrid_corners_in[r_i][c_i + 1].zs.empty() ||
              trigrid_centers_in[r_i][c_i].zs.empty()) {
            if (tgf_out[r_i][c_i][s_i].node_type != NONGROUND) {
              tgf_out[r_i][c_i][s_i].node_type = UNKNOWN;
            }
            continue;
          }
          refined_corner_1 =
              convertCornerToEigen(trigrid_corners_in[r_i + 1][c_i + 1]);
          refined_corner_2 =
              convertCornerToEigen(trigrid_corners_in[r_i][c_i + 1]);
          refined_center = convertCornerToEigen(trigrid_centers_in[r_i][c_i]);
          break;
        case 2:
          if (trigrid_corners_in[r_i][c_i + 1].zs.empty() ||
              trigrid_corners_in[r_i][c_i].zs.empty() ||
              trigrid_centers_in[r_i][c_i].zs.empty()) {
            if (tgf_out[r_i][c_i][s_i].node_type != NONGROUND) {
              tgf_out[r_i][c_i][s_i].node_type = UNKNOWN;
            }
            continue;
          }
          refined_corner_1 =
              convertCornerToEigen(trigrid_corners_in[r_i][c_i + 1]);
          refined_corner_2 = convertCornerToEigen(trigrid_corners_in[r_i][c_i]);
          refined_center = convertCornerToEigen(trigrid_centers_in[r_i][c_i]);
          break;
        case 3:
          if (trigrid_corners_in[r_i][c_i].zs.empty() ||
              trigrid_corners_in[r_i + 1][c_i].zs.empty() ||
              trigrid_centers_in[r_i][c_i].zs.empty()) {
            if (tgf_out[r_i][c_i][s_i].node_type != NONGROUND) {
              tgf_out[r_i][c_i][s_i].node_type = UNKNOWN;
            }
            continue;
          }
          refined_corner_1 = convertCornerToEigen(trigrid_corners_in[r_i][c_i]);
          refined_corner_2 =
              convertCornerToEigen(trigrid_corners_in[r_i + 1][c_i]);
          refined_center = convertCornerToEigen(trigrid_centers_in[r_i][c_i]);
          break;
        default:
          break;
        }

        // calculate the refined planar model in the node
        Eigen::Vector3f udpated_normal =
            (refined_corner_1 - refined_center)
                .cross(refined_corner_2 - refined_center);
        udpated_normal.normalize();

        if (udpated_normal(2, 0) < TH_NORMAL_) { // non-planar
          tgf_out[r_i][c_i][s_i].normal = udpated_normal;
          tgf_out[r_i][c_i][s_i].node_type = NONGROUND;
        } else {
          // planar
          Eigen::Vector3f updated_mean_pt;
          updated_mean_pt =
              (refined_corner_1 + refined_corner_2 + refined_center) / 3.0;

          tgf_out[r_i][c_i][s_i].normal = udpated_normal;
          tgf_out[r_i][c_i][s_i].mean_pt = updated_mean_pt;
          tgf_out[r_i][c_i][s_i].d = -(udpated_normal.dot(updated_mean_pt));
          tgf_out[r_i][c_i][s_i].th_dist_d =
              TH_DIST_ - tgf_out[r_i][c_i][s_i].d;
          tgf_out[r_i][c_i][s_i].th_outlier_d =
              -TH_OUTLIER_ - tgf_out[r_i][c_i][s_i].d;

          tgf_out[r_i][c_i][s_i].node_type = GROUND;
        }
      }
    }
  }

  return;
}

void TravelGroundFilterComponent::fitTGFWiseTraversableTerrainModel(
    TriGridField &tgf, std::vector<std::vector<TriGridCorner>> &trigrid_corners,
    std::vector<std::vector<TriGridCorner>> &trigrid_centers) {

  updateTGFCornersCenters(trigrid_corners, trigrid_centers);

  revertTraversableNodes(trigrid_corners, trigrid_centers, tgf);

  return;
}

void TravelGroundFilterComponent::segmentNodeGround(
    const TriGridNode &node_in, pcl::PointCloud<PointType> &node_ground_out,
    pcl::PointCloud<PointType> &node_nonground_out,
    pcl::PointCloud<PointType> &node_obstacle_out,
    pcl::PointCloud<PointType> &node_outlier_out) {
  node_ground_out.clear();
  node_nonground_out.clear();
  node_obstacle_out.clear();
  node_outlier_out.clear();

  // segment ground
  Eigen::MatrixXf points(node_in.ptCloud.points.size(), 3);
  int j = 0;
  for (auto &p : node_in.ptCloud.points) {
    points.row(j++) << p.x, p.y, p.z;
  }

  Eigen::VectorXf result =
      points * node_in.normal; // distances of each point from the plane
  for (int r = 0; r < result.rows(); r++) {
    if (result[r] < node_in.th_dist_d) {
      if (result[r] < node_in.th_outlier_d) {
        node_outlier_out.push_back(node_in.ptCloud.points[r]);
      } else {
        node_ground_out.push_back(node_in.ptCloud.points[r]);
      }
    } else {
      node_nonground_out.push_back(node_in.ptCloud.points[r]);
      if (result[r] < TH_OBSTACLE_HEIGHT_ - node_in.d) {
        node_obstacle_out.push_back(node_in.ptCloud.points[r]);
        node_obstacle_out.points.back().intensity = result[r] + node_in.d;
      }
    }
  }

  return;
}

void TravelGroundFilterComponent::segmentTGFGround(
    const TriGridField &tgf_in, pcl::PointCloud<PointType> &ground_cloud_out,
    pcl::PointCloud<PointType> &nonground_cloud_out,
    pcl::PointCloud<PointType> &obstacle_cloud_out,
    pcl::PointCloud<PointType> &outlier_cloud_out) {
  ground_cloud_out.clear();
  nonground_cloud_out.clear();
  obstacle_cloud_out.clear();

  for (int r_i = 0; r_i < rows_; r_i++) {
    for (int c_i = 0; c_i < cols_; c_i++) {
      for (int s_i = 0; s_i < tgf_in[r_i][c_i].size(); s_i++) {
        if (!tgf_in[r_i][c_i][s_i].is_curr_data) {
          continue;
        }
        if (tgf_in[r_i][c_i][s_i].node_type == GROUND) {
          segmentNodeGround(tgf_in[r_i][c_i][s_i], ptCloud_nodewise_ground_,
                            ptCloud_nodewise_nonground_,
                            ptCloud_nodewise_obstacle_,
                            ptCloud_nodewise_outliers_);
        } else {
          ptCloud_nodewise_nonground_ = tgf_in[r_i][c_i][s_i].ptCloud;
          ptCloud_nodewise_obstacle_ = tgf_in[r_i][c_i][s_i].ptCloud;
        }
        ground_cloud_out += ptCloud_nodewise_ground_;
        nonground_cloud_out += ptCloud_nodewise_nonground_;
        outlier_cloud_out += ptCloud_nodewise_outliers_;
        obstacle_cloud_out += ptCloud_nodewise_obstacle_;
      }
    }
  }

  return;
}

} // namespace ground_segmentation

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ground_segmentation::TravelGroundFilterComponent)
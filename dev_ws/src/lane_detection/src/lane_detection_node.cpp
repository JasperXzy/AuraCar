#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <foxglove_msgs/msg/image_annotations.hpp>
#include <foxglove_msgs/msg/points_annotation.hpp>
#include <foxglove_msgs/msg/point2.hpp>
#include <std_msgs/msg/header.hpp>
#include <opencv2/core.hpp>
#include <memory>
#include <chrono>
#include <vector>
#include <functional>
#include <cstdio>

#include "lane_detection/clrnet_infer.hpp"

namespace {
// 构建 ImageAnnotations
inline foxglove_msgs::msg::ImageAnnotations build_image_annotations(
    const std_msgs::msg::Header& header,
    uint32_t /*width*/, uint32_t /*height*/,  // 未使用，仅保留参数位
    const std::vector<std::vector<cv::Point2f>>& lanes) {
  		foxglove_msgs::msg::ImageAnnotations msg;
  		msg.points.resize(lanes.size());
  		for (size_t i = 0; i < lanes.size(); ++i) {
    		auto& pa = msg.points[i];
    		pa.timestamp = header.stamp;
    		pa.type = foxglove_msgs::msg::PointsAnnotation::LINE_STRIP;
    		pa.outline_color.r = 0.0f;
    		pa.outline_color.g = 1.0f;
    		pa.outline_color.b = 0.0f;
    		pa.outline_color.a = 1.0f;
    		pa.thickness = 2.0;
    		pa.points.reserve(lanes[i].size());
    			for (const auto& p : lanes[i]) {
      				foxglove_msgs::msg::Point2 pt;
      				pt.x = p.x;
      				pt.y = p.y;
      				pa.points.push_back(pt);
    			}
  		}
  	return msg;
	}
}  // namespace

class LaneDetectionNode : public rclcpp::Node {
public:
LaneDetectionNode() : rclcpp::Node("lane_detection") {
    // 参数声明
    this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    this->declare_parameter<std::string>("image_encoding", "nv12");
    this->declare_parameter<std::string>("annotations_topic", "/lane/annotations");
    this->declare_parameter<std::string>("model_path", "dev_ws/model/clrnet/clrnet.om");
    this->declare_parameter<int>("device_id", 0);
    this->declare_parameter<int>("input_width", 800);
    this->declare_parameter<int>("input_height", 320);
    this->declare_parameter<int>("pre_resize_height", 0);
    this->declare_parameter<bool>("use_dvpp", true);
    this->declare_parameter<bool>("use_aipp", true);
    this->declare_parameter<int>("S", 72);
    this->declare_parameter<int>("cut_height", 270);
    this->declare_parameter<int>("ori_height", 590);
    this->declare_parameter<double>("confidence_threshold", 0.3);
    this->declare_parameter<double>("nms_thres", 50.0);
    this->declare_parameter<int>("nms_topk", 5);
    this->declare_parameter<int>("horizon_y_px", -1);

    // 读取话题参数
    const std::string image_topic = this->get_parameter("image_topic").as_string();
    const std::string annotations_topic = this->get_parameter("annotations_topic").as_string();

    // Publisher
    annotations_pub_ = this->create_publisher<foxglove_msgs::msg::ImageAnnotations>(
        annotations_topic, rclcpp::QoS(10));

    // 组装推理选项
    ClrNetOptions opts;
    opts.model_path = this->get_parameter("model_path").as_string();
    opts.device_id = this->get_parameter("device_id").as_int();
    opts.input_width = this->get_parameter("input_width").as_int();
    opts.input_height = this->get_parameter("input_height").as_int();
    opts.pre_resize_height = this->get_parameter("pre_resize_height").as_int();
    opts.use_dvpp = this->get_parameter("use_dvpp").as_bool();
    opts.use_aipp = this->get_parameter("use_aipp").as_bool();
    opts.S = this->get_parameter("S").as_int();
    opts.cut_height = this->get_parameter("cut_height").as_int();
    opts.ori_height = this->get_parameter("ori_height").as_int();
    opts.confidence_threshold = static_cast<float>(this->get_parameter("confidence_threshold").as_double());
    opts.nms_thres = static_cast<float>(this->get_parameter("nms_thres").as_double());
    opts.nms_topk = this->get_parameter("nms_topk").as_int();
    opts.horizon_y_px = this->get_parameter("horizon_y_px").as_int();

    // 推理器初始化
    infer_ = std::make_unique<ClrNetInfer>(opts);
    if (!infer_->init()) {
      	RCLCPP_FATAL(this->get_logger(), "Failed to initialize ClrNetInfer");
      	throw std::runtime_error("ClrNet init failed");
    }

    // 订阅图像
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic, rclcpp::SensorDataQoS(),
        std::bind(&LaneDetectionNode::onImage, this, std::placeholders::_1));

    // 打印关键配置
    RCLCPP_INFO(this->get_logger(),
                "Cfg: model=%s dev=%d in=%dx%d pre_h=%d dvpp=%d aipp=%d S=%d cut=%d/%d conf=%.2f nms=%.1f topk=%d horizon=%d",
                opts.model_path.c_str(), opts.device_id, opts.input_width, opts.input_height,
                opts.pre_resize_height, opts.use_dvpp, opts.use_aipp, opts.S, opts.cut_height, opts.ori_height,
                opts.confidence_threshold, opts.nms_thres, opts.nms_topk, opts.horizon_y_px);
  }

private:
void onImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    using clock = std::chrono::steady_clock;
    static auto last_ts = clock::now();
    static size_t frame_cnt = 0;

    const auto t0 = clock::now();

    // 编码检查
    if (msg->encoding != "nv12") {
      	RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Unexpected encoding: %s", msg->encoding.c_str());
      	return;
    }

    std::vector<std::vector<cv::Point2f>> lanes;
    const bool ok = infer_->inferFromNV12(*msg, lanes);
    if (!ok) {
      	RCLCPP_WARN(this->get_logger(), "Inference failed for frame");
      	return;
    }

    // 发布到 Foxglove ImageAnnotations
    auto anno = build_image_annotations(msg->header, msg->width, msg->height, lanes);
    annotations_pub_->publish(anno);

    // 端到端耗时与简单 FPS
    const auto t1 = clock::now();
    const double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    frame_cnt++;
    const double dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - last_ts).count() / 1000.0;
    const double fps = dt > 0 ? frame_cnt / dt : 0.0;
    if (frame_cnt % 30 == 0) {
      	RCLCPP_INFO(this->get_logger(), "[Perf] e2e=%.3f ms | fps=%.2f | lanes=%zu", ms, fps, lanes.size());
      	last_ts = t1;
      	frame_cnt = 0;
    } else {
      	RCLCPP_DEBUG(this->get_logger(), "[Perf] e2e=%.3f ms | lanes=%zu", ms, lanes.size());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr annotations_pub_;
  std::unique_ptr<ClrNetInfer> infer_;
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaneDetectionNode>());
  rclcpp::shutdown();
  return 0;
}

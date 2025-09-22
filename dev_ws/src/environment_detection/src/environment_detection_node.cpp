#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <foxglove_msgs/msg/image_annotations.hpp>
#include <foxglove_msgs/msg/points_annotation.hpp>
#include <foxglove_msgs/msg/point2.hpp>
#include <std_msgs/msg/header.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include <vector>
#include <functional>
#include <cstdio>
#include <cstring>

#include "environment_detection/yolo11_seg_infer.hpp"

namespace {
// 构建 ImageAnnotations
inline foxglove_msgs::msg::ImageAnnotations build_image_annotations(
    const std_msgs::msg::Header& header,
    uint32_t /*width*/, uint32_t /*height*/,  // 未使用，仅保留参数位
    const std::vector<environment_detection::BoundBox>& boxes,
    const std::vector<std::string>& classNames) {
    foxglove_msgs::msg::ImageAnnotations msg;
    msg.points.resize(boxes.size());
    
    // 定义颜色映射
    const std::vector<std::vector<float>> colors = {
        {1.0f, 0.0f, 0.0f},  // 红色
        {0.0f, 1.0f, 0.0f},  // 绿色
        {0.0f, 0.0f, 1.0f},  // 蓝色
        {1.0f, 1.0f, 0.0f},  // 黄色
        {1.0f, 0.0f, 1.0f},  // 品红色
        {0.0f, 1.0f, 1.0f},  // 青色
    };
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        auto& pa = msg.points[i];
        pa.timestamp = header.stamp;
        pa.type = foxglove_msgs::msg::PointsAnnotation::LINE_LOOP;
        
        // 设置颜色
        auto color = colors[i % colors.size()];
        pa.outline_color.r = color[0];
        pa.outline_color.g = color[1];
        pa.outline_color.b = color[2];
        pa.outline_color.a = 1.0f;
        pa.thickness = 2.0;
        
        // 构建边界框的四个顶点
        const auto& box = boxes[i];
        float x1 = box.x - box.width / 2;
        float y1 = box.y - box.height / 2;
        float x2 = box.x + box.width / 2;
        float y2 = box.y + box.height / 2;
        
        pa.points.resize(4);
        pa.points[0].x = x1; pa.points[0].y = y1;  // 左上
        pa.points[1].x = x2; pa.points[1].y = y1;  // 右上
        pa.points[2].x = x2; pa.points[2].y = y2;  // 右下
        pa.points[3].x = x1; pa.points[3].y = y2;  // 左下
    }
    
    return msg;
}
}  // namespace

class EnvironmentDetectionNode : public rclcpp::Node {
public:
    EnvironmentDetectionNode() : rclcpp::Node("environment_detection") {
        // 参数声明
        this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
        this->declare_parameter<std::string>("image_encoding", "nv12");
        this->declare_parameter<std::string>("annotations_topic", "/environment/annotations");
        this->declare_parameter<std::string>("segmentation_topic", "/environment/segmentation");
        this->declare_parameter<std::string>("model_path", "model/yolo11n-seg.om");
        
        this->declare_parameter<bool>("use_dvpp", true);
        this->declare_parameter<bool>("use_aipp", true);
        this->declare_parameter<int>("device_id", 0);
        
        this->declare_parameter<int>("input_width", 640);
        this->declare_parameter<int>("input_height", 640);
        
        this->declare_parameter<std::string>("qos_history", "keep_last");
        this->declare_parameter<int>("qos_depth", 2);
        
        this->declare_parameter<float>("confidence_threshold", 0.35);
        this->declare_parameter<float>("nms_threshold", 0.45);
        
        // 获取参数
        auto image_topic = this->get_parameter("image_topic").as_string();
        image_encoding_ = this->get_parameter("image_encoding").as_string();
        auto annotations_topic = this->get_parameter("annotations_topic").as_string();
        auto segmentation_topic = this->get_parameter("segmentation_topic").as_string();
        auto model_path = this->get_parameter("model_path").as_string();
        
        use_dvpp_ = this->get_parameter("use_dvpp").as_bool();
        use_aipp_ = this->get_parameter("use_aipp").as_bool();
        device_id_ = this->get_parameter("device_id").as_int();
        
        input_width_ = this->get_parameter("input_width").as_int();
        input_height_ = this->get_parameter("input_height").as_int();
        
        auto qos_history_str = this->get_parameter("qos_history").as_string();
        auto qos_depth = this->get_parameter("qos_depth").as_int();
        
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
        nms_threshold_ = this->get_parameter("nms_threshold").as_double();
        
        // 设置QoS
        rmw_qos_profile_t qos_profile = rmw_qos_profile_default;
        if (qos_history_str == "keep_last") {
            qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
        } else if (qos_history_str == "keep_all") {
            qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_ALL;
        }
        qos_profile.depth = qos_depth;
        
        // 初始化推理引擎
        segInfer_ = std::make_unique<environment_detection::YOLO11SegInfer>(
            model_path, input_width_, input_height_, 
            confidence_threshold_, nms_threshold_);
        
        if (segInfer_->InitResource() != 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize YOLO11SegInfer");
            rclcpp::shutdown();
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "YOLO11SegInfer initialized successfully");
        
        // 创建订阅者和发布者
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic, rclcpp::SensorDataQoS(),
            std::bind(&EnvironmentDetectionNode::image_callback, this, std::placeholders::_1));
        
        annotations_publisher_ = this->create_publisher<foxglove_msgs::msg::ImageAnnotations>(
            annotations_topic, rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos_profile)));
        
        segmentation_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            segmentation_topic, rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos_profile)));
        
        RCLCPP_INFO(this->get_logger(), "Environment Detection Node initialized");
        RCLCPP_INFO(this->get_logger(), "  - Image topic: %s", image_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - Annotations topic: %s", annotations_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - Segmentation topic: %s", segmentation_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "  - Model path: %s", model_path.c_str());
        RCLCPP_INFO(this->get_logger(), "  - Input size: %dx%d", input_width_, input_height_);
        RCLCPP_INFO(this->get_logger(), "  - Confidence threshold: %.2f", confidence_threshold_);
        RCLCPP_INFO(this->get_logger(), "  - NMS threshold: %.2f", nms_threshold_);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // 编码检查
            if (msg->encoding != "nv12") {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                   "Unexpected encoding: %s", msg->encoding.c_str());
                return;
            }
            
            // 检查图像尺寸
            if (msg->width != 1280 || msg->height != 720) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                   "Unexpected image size: %dx%d, expected 1280x720", 
                                   msg->width, msg->height);
            }
            
            // 直接使用推理引擎处理NV12格式
            environment_detection::SegmentationResult result;
            if (segInfer_->ProcessImageFromNV12(*msg, result) != 0) {
                RCLCPP_ERROR(this->get_logger(), "Failed to process NV12 image");
                return;
            }
            
            // 发布检测注释
            if (!result.boxes.empty()) {
                auto annotations_msg = build_image_annotations(msg->header, msg->width, msg->height,
                                                             result.boxes, segInfer_->GetClassNames());
                annotations_publisher_->publish(annotations_msg);
                
                // 记录检测结果
                for (const auto& box : result.boxes) {
                    if (box.classIndex < segInfer_->GetClassNames().size()) {
                        RCLCPP_DEBUG(this->get_logger(), 
                                  "Detected: %s (%.2f confidence) at [%.1f, %.1f, %.1f, %.1f]",
                                  segInfer_->GetClassNames()[box.classIndex].c_str(),
                                  box.score, box.x, box.y, box.width, box.height);
                    }
                }
            }
            
            // 发布分割掩码
            if (!result.combinedMask.empty()) {
                auto segmentation_msg = std::make_shared<sensor_msgs::msg::Image>();
                segmentation_msg->header = msg->header;
                segmentation_msg->height = result.combinedMask.rows;
                segmentation_msg->width = result.combinedMask.cols;
                segmentation_msg->encoding = "mono8";
                segmentation_msg->is_bigendian = false;
                segmentation_msg->step = result.combinedMask.cols;
                
                size_t data_size = segmentation_msg->step * segmentation_msg->height;
                segmentation_msg->data.resize(data_size);
                std::memcpy(segmentation_msg->data.data(), result.combinedMask.data, data_size);
                
                segmentation_publisher_->publish(*segmentation_msg);
            }
            
            // 计算处理时间和FPS
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double ms = duration.count() / 1000.0;
            
            static auto last_ts = std::chrono::steady_clock::now();
            static size_t frame_cnt = 0;
            frame_cnt++;
            
            auto current_time = std::chrono::steady_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_ts).count() / 1000.0;
            double fps = dt > 0 ? frame_cnt / dt : 0.0;
            
            if (frame_cnt % 30 == 0) {
                RCLCPP_INFO(this->get_logger(), "[Perf] e2e=%.3f ms | fps=%.2f | objects=%zu", 
                           ms, fps, result.boxes.size());
                last_ts = current_time;
                frame_cnt = 0;
            } else {
                RCLCPP_DEBUG(this->get_logger(), "[Perf] e2e=%.3f ms | objects=%zu", 
                            ms, result.boxes.size());
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in image callback: %s", e.what());
        }
    }

private:
    // ROS2 组件
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr annotations_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmentation_publisher_;
    
    // 推理引擎
    std::unique_ptr<environment_detection::YOLO11SegInfer> segInfer_;
    
    // 参数
    std::string image_encoding_;
    bool use_dvpp_;
    bool use_aipp_;
    int device_id_;
    int input_width_;
    int input_height_;
    double confidence_threshold_;
    double nms_threshold_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<EnvironmentDetectionNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("environment_detection"), 
                     "Exception in main: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}

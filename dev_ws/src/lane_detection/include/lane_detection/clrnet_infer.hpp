#pragma once
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

// 推理与后处理的可配置项
struct ClrNetOptions {
	std::string model_path;
	int device_id{0};
	int input_width{800};      // 模型逻辑输入宽
	int input_height{320};     // 模型逻辑输入高
	int pre_resize_height{0};
	bool use_dvpp{true};
	bool use_aipp{true};
	int S{72};                 // 条带数
	int cut_height{270};       // 训练/预处理的裁剪像素
	int ori_height{590};       // 训练原图高度
	float confidence_threshold{0.5f};
	float nms_thres{50.0f};
	int nms_topk{5};
	int horizon_y_px{-1};      // 发布图像像素坐标的地平线：<0 关闭，>=0 时丢弃其以上的点
};

// Ascend 初始化、DVPP 预处理、模型推理、以及调用后处理生成车道点
class ClrNetInfer {
public:
	explicit ClrNetInfer(const ClrNetOptions& opts);
	~ClrNetInfer();
	bool init();
	bool inferFromNV12(const sensor_msgs::msg::Image& nv12, std::vector<std::vector<cv::Point2f>>& lanes);
private:
	struct Impl;
	Impl* impl;
};

#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <cstdint>

// 后处理可配置项
struct ClrNetPostOptions {
	int S{72};            // 条带数
	int imgW{800};        // 模型输入宽
	int imgH{320};        // 模型输入高
	int cutHeight{270};   // 训练/预处理裁剪的像素
	int oriH{590};        // 训练原图高度
	// 地平线像素 y（发布图像坐标系），>=0 时丢弃其以上的点，避免消失点以远的伪线
	int horizonYPx{-1};
	float confThresh{0.5f};  // 置信度阈值
	float nmsThres{50.0f};   // NMS 阈值
	int nmsTopk{5};          // NMS 后保留的最大车道条数
};

// 后处理器：解码 -> NMS -> 映射到输出尺寸
class ClrNetPostProcessor {
public:
	explicit ClrNetPostProcessor(const ClrNetPostOptions& opt);
	std::vector<std::vector<cv::Point2f>> decode(float* pred, int outImgW, int outImgH);
private:
	struct Lane {
	float conf{0};
	float start_y{0};
	float start_x{0};
	float theta{0};
	float length{0};
	std::vector<float> offsets;
	};
	// 二分类 softmax 概率
	float softmax2(float a, float b) const;
	// 将模型输出解码为候选车道列表
	std::vector<Lane> decodeLanes(float* pred) const;
	// 计算两条车道在重叠条带区间上的平均横向像素距离
	float laneDist(const Lane& A, const Lane& B) const;
	// 简单 NMS
	std::vector<Lane> nms(std::vector<Lane>& lanes) const;
	// 将归一化坐标转为发布图像像素坐标
	std::vector<std::vector<cv::Point2f>> toPoints(const std::vector<Lane>& lanes, int outImgW, int outImgH) const;
private:
	ClrNetPostOptions opt_;
	int nStrips_;
	int nOffsets_;
	std::vector<float> anchorYs_;
};

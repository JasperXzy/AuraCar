#include "lane_detection/postprocess.hpp"
#include <algorithm>
#include <cmath>

// 根据 S 初始化条带锚点 anchorYs
ClrNetPostProcessor::ClrNetPostProcessor(const ClrNetPostOptions& opt)
	: opt_(opt) {
	nStrips_ = opt_.S - 1;
	nOffsets_ = opt_.S;
	anchorYs_.reserve(nOffsets_);
	for (int i = 0; i < nOffsets_; ++i) {
		anchorYs_.push_back(1.0f - static_cast<float>(i) / nStrips_);
	}
}

// 二分类 softmax，返回第二个分量（正类）的概率
float ClrNetPostProcessor::softmax2(float a, float b) const {
	float m = std::max(a, b);
	float ea = std::exp(a - m);
	float eb = std::exp(b - m);
	return eb / (ea + eb);
}

// 解析模型输出为候选车道列表
std::vector<ClrNetPostProcessor::Lane> ClrNetPostProcessor::decodeLanes(float* pred) const {
	std::vector<Lane> lanes;
	const int numLanes = 192;
	const int laneDim = 78;
	for (int i = 0; i < numLanes; ++i) {
		float* p = pred + i * laneDim;
		float conf = softmax2(p[0], p[1]);
		if (conf < opt_.confThresh) continue;
		Lane lane;
		lane.conf = conf;
		lane.start_y = p[2];
		lane.start_x = p[3];
		lane.theta = p[4];
		lane.length = p[5];
		lane.offsets.reserve(nOffsets_);
		for (int k = 0; k < nOffsets_; ++k) lane.offsets.push_back(p[6 + k]);
		lanes.push_back(std::move(lane));
	}
	std::sort(lanes.begin(), lanes.end(), [](const Lane& a, const Lane& b){ return a.conf > b.conf; });
	return lanes;
}

// 计算两条车道在重叠条带区间上的平均横向像素距离
float ClrNetPostProcessor::laneDist(const Lane& A, const Lane& B) const {
	int startA = static_cast<int>(A.start_y * nStrips_ + 0.5f);
	int startB = static_cast<int>(B.start_y * nStrips_ + 0.5f);
	int start = std::max(startA, startB);
	int endA = startA + static_cast<int>(A.length * nStrips_ + 0.5f) - 1;
	int endB = startB + static_cast<int>(B.length * nStrips_ + 0.5f) - 1;
	int end = std::min(std::min(endA, endB), nStrips_);
	if (end < start) return 1e9f;
	float acc = 0.0f; int cnt = 0;
	for (int i = start; i <= end; ++i) {
		if (i < static_cast<int>(A.offsets.size()) && i < static_cast<int>(B.offsets.size())) {
			acc += std::abs((A.offsets[i] - B.offsets[i]) * (opt_.imgW - 1));
			++cnt;
		}
	}
	return cnt > 0 ? acc / cnt : 1e9f;
}

// 简单 NMS，保留置信度高的车道，移除平均距离小于阈值的相似车道
std::vector<ClrNetPostProcessor::Lane> ClrNetPostProcessor::nms(std::vector<Lane>& lanes) const {
	std::vector<bool> removed(lanes.size(), false);
	std::vector<Lane> keep;
	for (size_t i = 0; i < lanes.size(); ++i) {
		if (removed[i]) continue;
		keep.push_back(lanes[i]);
		for (size_t j = i + 1; j < lanes.size(); ++j) {
			if (removed[j]) continue;
			if (laneDist(lanes[i], lanes[j]) < opt_.nmsThres) removed[j] = true;
		}
	}
	if (keep.size() > static_cast<size_t>(opt_.nmsTopk)) keep.resize(opt_.nmsTopk);
	return keep;
}

// 将归一化坐标转为发布图像像素坐标
// - y 轴补偿：考虑训练/预处理的顶部裁剪（cutHeight/oriH）
// - 地平线裁剪：opt_.horizonYPx >= 0 时，丢弃其以上的点
std::vector<std::vector<cv::Point2f>> ClrNetPostProcessor::toPoints(const std::vector<Lane>& lanes, int outImgW, int outImgH) const {
	std::vector<std::vector<cv::Point2f>> res;
	res.reserve(lanes.size());
	for (const auto& lane : lanes) {
		int start = static_cast<int>(lane.start_y * nStrips_ + 0.5f);
		int end = start + static_cast<int>(lane.length * nStrips_ + 0.5f) - 1;
		end = std::min(end, nStrips_);
		std::vector<cv::Point2f> pts;
		for (int i = start; i <= end; ++i) {
			if (i < static_cast<int>(lane.offsets.size())) {
	float factor = static_cast<float>(opt_.cutHeight) / static_cast<float>(opt_.oriH);
	float y_norm = (1.0f - factor) * anchorYs_[i] + factor;
				float x_norm = lane.offsets[i];
				float x = x_norm * (outImgW - 1);
				float y = y_norm * (outImgH - 1);
				if (opt_.horizonYPx >= 0 && y < static_cast<float>(opt_.horizonYPx)) {
					continue;
				}
				pts.emplace_back(x, y);
			}
		}
		if (!pts.empty()) res.push_back(std::move(pts));
	}
	return res;
}

// 解码 -> NMS -> 像素坐标
std::vector<std::vector<cv::Point2f>> ClrNetPostProcessor::decode(float* pred, int outImgW, int outImgH) {
	auto lanes = decodeLanes(pred);
	auto kept = nms(lanes);
	return toPoints(kept, outImgW, outImgH);
}

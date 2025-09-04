#include "lane_detection/clrnet_infer.hpp"
#include "lane_detection/postprocess.hpp"
#include "AclLiteModel.h"
#include "AclLiteResource.h"
#include "AclLiteImageProc.h"
#include "AclLiteUtils.h"
#include "AclLiteError.h"
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <algorithm>

struct ClrNetInfer::Impl {
	ClrNetOptions opts;
	AclLiteResource acl;
	AclLiteImageProc dvpp;
	AclLiteModel model;
	bool inited{false};

	Impl(const ClrNetOptions& o) : opts(o) {}

	bool init() {
		if (acl.Init() != ACLLITE_OK) return false;
		if (dvpp.Init() != ACLLITE_OK) return false;
		if (model.Init(opts.model_path) != ACLLITE_OK) return false;
		inited = true;
		return true;
	}

	bool inferFromNV12(const sensor_msgs::msg::Image& nv12, std::vector<std::vector<cv::Point2f>>& lanes) {
		if (!inited) return false;
	auto t_begin = std::chrono::steady_clock::now();
		// 将 ROS Image 包装为 ImageData (NV12)
		ImageData input{};
		input.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
		input.width = nv12.width;
		input.height = nv12.height;
	// NV12 通常 width 按16对齐，height 按2对齐
	auto alignUp = [](uint32_t x, uint32_t a) { return (x + a - 1) / a * a; };
	input.alignWidth = alignUp(input.width, 16);
	input.alignHeight = alignUp(input.height, 2);
		input.size = static_cast<uint32_t>(nv12.data.size());
		input.data = std::shared_ptr<uint8_t>(const_cast<uint8_t*>(nv12.data.data()), [](uint8_t*){});

		// 将输入从 Host 拷贝到 DVPP 内存
		ImageData inputDvpp{};
		aclrtRunMode runMode;
		(void)aclrtGetRunMode(&runMode);
		auto t_copy0 = std::chrono::steady_clock::now();
		if (CopyImageToDevice(inputDvpp, input, runMode, MEMORY_DVPP) != ACLLITE_OK) {
			return false;
		}
		auto t_copy1 = std::chrono::steady_clock::now();

	// DVPP Resize 到模型前置尺寸
	ImageData resized{};
		auto t_resize0 = std::chrono::steady_clock::now();
	int dvpp_h = opts.pre_resize_height > 0 ? opts.pre_resize_height : opts.input_height;
	if (dvpp.Resize(resized, inputDvpp, opts.input_width, dvpp_h) != ACLLITE_OK) {
			return false;
		}
		auto t_resize1 = std::chrono::steady_clock::now();

	// AIPP 在 OM 内配置，为避免 Execute 内部隐式拷贝，这里显式拷贝到 DEVICE 内存
	ImageData modelInput{};
	auto t_create_in0 = std::chrono::steady_clock::now();
	if (CopyImageToDevice(modelInput, resized, runMode, MEMORY_DEVICE) != ACLLITE_OK) return false;
	std::vector<InferenceOutput> outputs;
	if (model.CreateInput(modelInput.data.get(), modelInput.size) != ACLLITE_OK) return false;
		auto t_create_in1 = std::chrono::steady_clock::now();
		auto t_exec0 = t_create_in1;
		if (model.Execute(outputs) != ACLLITE_OK) return false;
		auto t_exec1 = std::chrono::steady_clock::now();

	// 后处理
	if (outputs.empty() || !outputs[0].data) return false;
	auto* pred = static_cast<float*>(outputs[0].data.get());
	ClrNetPostOptions pOpt;
	pOpt.S = opts.S;
	pOpt.imgW = opts.input_width;
	pOpt.imgH = opts.input_height;
	pOpt.cutHeight = opts.cut_height;
	pOpt.oriH = opts.ori_height;
	pOpt.confThresh = opts.confidence_threshold;
	pOpt.nmsThres = opts.nms_thres;
	pOpt.nmsTopk = opts.nms_topk;
	pOpt.horizonYPx = opts.horizon_y_px;
	ClrNetPostProcessor post(pOpt);
	auto t_post0 = std::chrono::steady_clock::now();
	lanes = post.decode(pred, static_cast<int>(nv12.width), static_cast<int>(nv12.height));
	auto t_post1 = std::chrono::steady_clock::now();

	// 输出性能统计
	auto ms = [](auto d){ return std::chrono::duration_cast<std::chrono::microseconds>(d).count() / 1000.0; };
	double t_copy_ms = ms(t_copy1 - t_copy0);
	double t_resize_ms = ms(t_resize1 - t_resize0);
	double t_create_in_ms = ms(t_create_in1 - t_create_in0);
	double t_exec_ms = ms(t_exec1 - t_exec0);
	double t_post_ms = ms(t_post1 - t_post0);
	double t_total_ms = ms(t_post1 - t_begin);
	double t_pre_ms = t_copy_ms + t_resize_ms + t_create_in_ms;
	ACLLITE_LOG_INFO("[Perf] infer frame: total=%.3f ms | preprocess=%.3f ms | execute=%.3f ms | postprocess=%.3f ms",
					 t_total_ms, t_pre_ms, t_exec_ms, t_post_ms);
		return true;
	}
};

ClrNetInfer::ClrNetInfer(const ClrNetOptions& opts) : impl(new Impl(opts)) {}
ClrNetInfer::~ClrNetInfer(){ delete impl; }
bool ClrNetInfer::init(){ return impl->init(); }
bool ClrNetInfer::inferFromNV12(const sensor_msgs::msg::Image& nv12, std::vector<std::vector<cv::Point2f>>& lanes){
	return impl->inferFromNV12(nv12, lanes);
}

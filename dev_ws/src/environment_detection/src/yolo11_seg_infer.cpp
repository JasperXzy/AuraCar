#include "environment_detection/yolo11_seg_infer.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>

namespace environment_detection {

// 环境检测类别标签（可根据模型调整）
const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

YOLO11SegInfer::YOLO11SegInfer(const std::string& modelPath,
                               int modelWidth, int modelHeight,
                               float confidenceThreshold, float nmsThreshold)
    : modelPath_(modelPath)
    , modelWidth_(modelWidth)
    , modelHeight_(modelHeight)
    , confidenceThreshold_(confidenceThreshold)
    , nmsThreshold_(nmsThreshold)
    , classNames_(CLASS_NAMES)
    , initialized_(false) {
}

YOLO11SegInfer::~YOLO11SegInfer() {
    ReleaseResource();
}

int YOLO11SegInfer::InitResource() {
    if (initialized_) {
        return 0;
    }

    // 初始化ACL资源
    AclLiteError ret = aclResource_.Init();
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("ACL resource init failed, errorCode is %d", ret);
        return -1;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return -1;
    }

    // 初始化DVPP资源
    ret = imageProcess_.Init();
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return -1;
    }

    // 加载模型
    ret = model_.Init(modelPath_.c_str());
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return -1;
    }

    initialized_ = true;
    ACLLITE_LOG_INFO("YOLO11SegInfer init success");
    return 0;
}

void YOLO11SegInfer::ReleaseResource() {
    if (initialized_) {
        model_.DestroyResource();
        imageProcess_.DestroyResource();
        aclResource_.Release();
        initialized_ = false;
    }
}

int YOLO11SegInfer::ProcessImageFromNV12(const sensor_msgs::msg::Image& msg, SegmentationResult& result) {
    if (!initialized_) {
        ACLLITE_LOG_ERROR("YOLO11SegInfer not initialized");
        return -1;
    }

    if (msg.encoding != "nv12") {
        ACLLITE_LOG_ERROR("Expected NV12 encoding, got: %s", msg.encoding.c_str());
        return -1;
    }

    // 预处理NV12图像
    if (PreprocessImageFromNV12(msg) != 0) {
        ACLLITE_LOG_ERROR("Preprocess NV12 image failed");
        return -1;
    }

    // 执行推理
    std::vector<InferenceOutput> inferOutputs;
    if (DoInference(inferOutputs) != 0) {
        ACLLITE_LOG_ERROR("Inference failed");
        return -1;
    }

    // 后处理结果
    if (PostprocessResult(inferOutputs, msg.width, msg.height, result) != 0) {
        ACLLITE_LOG_ERROR("Postprocess failed");
        return -1;
    }

    return 0;
}

int YOLO11SegInfer::PreprocessImageFromNV12(const sensor_msgs::msg::Image& msg) {
    // 将NV12数据拷贝到设备内存
    ImageData srcImage;
    srcImage.width = msg.width;
    srcImage.height = msg.height;
    srcImage.alignWidth = msg.width;
    srcImage.alignHeight = msg.height;
    srcImage.size = msg.data.size();
    
    // 分配设备内存并拷贝NV12数据
    AclLiteError ret = CopyDataToDevice(srcImage.data, msg.data.data(), srcImage.size, runMode_, MEMORY_DVPP);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("Copy NV12 data to device failed, errorCode is %d", ret);
        return -1;
    }

    // 调整图像尺寸到模型输入尺寸
    // 对于NV12格式，直接resize，AIPP会处理颜色转换和归一化
    ret = imageProcess_.Resize(resizedImage_, srcImage, modelWidth_, modelHeight_);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("Resize NV12 image failed, errorCode is %d", ret);
        return -1;
    }

    return 0;
}

int YOLO11SegInfer::DoInference(std::vector<InferenceOutput>& inferOutputs) {
    // 创建模型输入
    AclLiteError ret = model_.CreateInput(static_cast<void*>(resizedImage_.data.get()), resizedImage_.size);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return -1;
    }

    // 执行推理
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("Execute model failed, errorCode is %d", ret);
        return -1;
    }

    return 0;
}

int YOLO11SegInfer::PostprocessResult(const std::vector<InferenceOutput>& inferOutputs,
                                      int srcWidth, int srcHeight,
                                      SegmentationResult& result) {
    if (inferOutputs.empty()) {
        ACLLITE_LOG_ERROR("Inference outputs are empty");
        return -1;
    }

    // YOLO11分割模型输出结构:
    // output0: [1, 4+classNum+32, 8400] - 检测结果 (4坐标 + classNum类别 + 32掩码系数)
    // output1: [1, 32, 160, 160] - 掩码原型
    uint32_t outputDataBufId = 0;
    uint32_t maskProtoBufId = 1;

    float* classBuff = static_cast<float*>(inferOutputs[outputDataBufId].data.get());
    float* maskProtoBuff = nullptr;

    // 检查是否有掩码原型输出
    if (inferOutputs.size() > 1) {
        maskProtoBuff = static_cast<float*>(inferOutputs[maskProtoBufId].data.get());
    }

    // 类别数量
    size_t classNum = classNames_.size();
    // 坐标数量 (x, y, width, height)
    size_t offset = 4;
    // 检测框总数 YOLO11: [1,4+classNum+32,8400]
    size_t modelOutputBoxNum = 8400;
    // 掩码原型数量
    size_t maskProtoNum = 32;

    // 按置信度过滤边界框
    std::vector<BoundBox> boxes;
    size_t yIndex = 1;
    size_t widthIndex = 2;
    size_t heightIndex = 3;

    for (size_t i = 0; i < modelOutputBoxNum; ++i) {
        float maxValue = 0;
        size_t maxIndex = 0;
        
        // 找到最大置信度的类别
        for (size_t j = 0; j < classNum; ++j) {
            float value = classBuff[(offset + j) * modelOutputBoxNum + i];
            if (value > maxValue) {
                maxIndex = j;
                maxValue = value;
            }
        }

        if (maxValue > confidenceThreshold_) {
            BoundBox box;
            // 将坐标转换到原始图像尺寸
            box.x = classBuff[i] * srcWidth / modelWidth_;
            box.y = classBuff[yIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.width = classBuff[widthIndex * modelOutputBoxNum + i] * srcWidth / modelWidth_;
            box.height = classBuff[heightIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.score = maxValue;
            box.classIndex = maxIndex;
            box.index = i;

            // 提取掩码系数 (从第5个位置开始，共32个系数)
            size_t maskCoeffStart = offset + classNum; // 4 + classNum
            for (size_t k = 0; k < maskProtoNum; k++) {
                float coeff = classBuff[(maskCoeffStart + k) * modelOutputBoxNum + i];
                box.mask.push_back(coeff);
            }

            boxes.push_back(box);
        }
    }

    ACLLITE_LOG_INFO("Filtered boxes by confidence threshold > %f, found %lu boxes", 
                     confidenceThreshold_, boxes.size());

    // 应用NMS
    std::vector<BoundBox> nmsResult = ApplyNMS(boxes);
    
    ACLLITE_LOG_INFO("Filtered boxes by NMS threshold > %f, final %lu boxes", 
                     nmsThreshold_, nmsResult.size());

    // 处理结果
    result.boxes = nmsResult;
    result.combinedMask = cv::Mat::zeros(srcHeight, srcWidth, CV_8UC1);
    result.classMasks.clear();
    result.classMasks.resize(classNum);

    for (size_t i = 0; i < classNum; ++i) {
        result.classMasks[i] = cv::Mat::zeros(srcHeight, srcWidth, CV_8UC1);
    }

    // 处理每个检测框的掩码
    for (const auto& box : nmsResult) {
        cv::Mat mask = ProcessMask(box, maskProtoBuff, srcWidth, srcHeight);
        
        // 合并到组合掩码
        cv::bitwise_or(result.combinedMask, mask, result.combinedMask);
        
        // 合并到对应类别的掩码
        if (box.classIndex < classNum) {
            cv::bitwise_or(result.classMasks[box.classIndex], mask, result.classMasks[box.classIndex]);
        }
    }

    return 0;
}

std::vector<BoundBox> YOLO11SegInfer::ApplyNMS(const std::vector<BoundBox>& boxes) {
    std::vector<BoundBox> result;
    std::vector<BoundBox> sortedBoxes = boxes;
    
    // 按置信度排序
    std::sort(sortedBoxes.begin(), sortedBoxes.end(), 
              [](const BoundBox& a, const BoundBox& b) { return a.score > b.score; });
    
    int32_t maxLength = std::max(modelWidth_, modelHeight_);
    
    while (!sortedBoxes.empty()) {
        result.push_back(sortedBoxes[0]);
        
        auto it = sortedBoxes.begin() + 1;
        while (it != sortedBoxes.end()) {
            BoundBox boxMax = sortedBoxes[0];
            BoundBox boxCompare = *it;
            
            // 为了避免不同类别的框相互干扰，按类别平移坐标点
            boxMax.x += maxLength * boxMax.classIndex;
            boxMax.y += maxLength * boxMax.classIndex;
            boxCompare.x += maxLength * boxCompare.classIndex;
            boxCompare.y += maxLength * boxCompare.classIndex;
            
            float iou = CalculateIoU(boxMax, boxCompare);
            
            // 按NMS阈值过滤框
            if (iou > nmsThreshold_) {
                it = sortedBoxes.erase(it);
            } else {
                ++it;
            }
        }
        
        sortedBoxes.erase(sortedBoxes.begin());
    }
    
    return result;
}

float YOLO11SegInfer::CalculateIoU(const BoundBox& box1, const BoundBox& box2) {
    // 计算两个框的重叠部分
    float xLeft = std::max(box1.x, box2.x);
    float yTop = std::max(box1.y, box2.y);
    float xRight = std::min(box1.x + box1.width, box2.x + box2.width);
    float yBottom = std::min(box1.y + box1.height, box2.y + box2.height);
    
    float width = std::max(0.0f, xRight - xLeft);
    float height = std::max(0.0f, yBottom - yTop);
    float intersectionArea = width * height;
    
    float unionArea = box1.width * box1.height + box2.width * box2.height - intersectionArea;
    
    return (unionArea > 0) ? (intersectionArea / unionArea) : 0.0f;
}

cv::Mat YOLO11SegInfer::ProcessMask(const BoundBox& box, const float* maskProtoData, 
                                    int srcWidth, int srcHeight) {
    cv::Mat mask = cv::Mat::zeros(srcHeight, srcWidth, CV_8UC1);
    
    // 如果没有掩码原型数据，使用简化的椭圆掩码
    if (maskProtoData == nullptr || box.mask.empty()) {
        // 计算检测框在原始图像中的位置
        int x1 = std::max(0, (int)(box.x - box.width / 2));
        int y1 = std::max(0, (int)(box.y - box.height / 2));
        int x2 = std::min(srcWidth, (int)(box.x + box.width / 2));
        int y2 = std::min(srcHeight, (int)(box.y + box.height / 2));
        
        // 创建椭圆掩码
        cv::Point center((x1 + x2) / 2, (y1 + y2) / 2);
        cv::Size axes((x2 - x1) / 2, (y2 - y1) / 2);
        cv::ellipse(mask, center, axes, 0, 0, 360, cv::Scalar(255), -1);
        
        // 使用高斯模糊使掩码更自然
        cv::GaussianBlur(mask, mask, cv::Size(9, 9), 0);
        return mask;
    }
    
    // 使用掩码系数和掩码原型生成真实的分割掩码
    // 掩码原型的尺寸通常是160x160
    int protoSize = 160;
    cv::Mat protoMask = cv::Mat::zeros(protoSize, protoSize, CV_32F);
    
    // 掩码系数和掩码原型的线性组合
    for (int y = 0; y < protoSize; y++) {
        for (int x = 0; x < protoSize; x++) {
            float sum = 0.0f;
            for (size_t k = 0; k < box.mask.size() && k < 32; k++) {
                // 掩码原型数据索引: [k * protoSize * protoSize + y * protoSize + x]
                int protoIdx = k * protoSize * protoSize + y * protoSize + x;
                sum += box.mask[k] * maskProtoData[protoIdx];
            }
            protoMask.at<float>(y, x) = sum;
        }
    }
    
    // 应用sigmoid激活函数
    cv::Mat sigmoidMask;
    cv::exp(-protoMask, sigmoidMask);
    sigmoidMask = 1.0 / (1.0 + sigmoidMask);
    
    // 将掩码调整到原始图像尺寸
    cv::Mat resizedMask;
    cv::resize(sigmoidMask, resizedMask, cv::Size(srcWidth, srcHeight));
    
    // 转换为8位图像
    cv::Mat mask8u;
    resizedMask.convertTo(mask8u, CV_8UC1, 255.0);
    
    // 应用阈值
    cv::threshold(mask8u, mask, 128, 255, cv::THRESH_BINARY);
    
    // 应用形态学操作平滑边缘
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    return mask;
}

} // namespace environment_detection

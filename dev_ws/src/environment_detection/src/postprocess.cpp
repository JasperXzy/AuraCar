#include "environment_detection/yolo11_seg_infer.hpp"

namespace environment_detection {

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

} // namespace environment_detection

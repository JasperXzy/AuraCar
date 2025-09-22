#ifndef YOLO11_SEG_INFER_HPP
#define YOLO11_SEG_INFER_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"

namespace environment_detection {

// 边界框结构体，包含分割掩码信息
struct BoundBox {
    float x, y, width, height;     // 边界框坐标和尺寸
    float score;                   // 置信度
    size_t classIndex;             // 类别索引
    size_t index;                  // 原始索引
    std::vector<float> mask;       // 掩码系数
};

// 分割结果结构体
struct SegmentationResult {
    std::vector<BoundBox> boxes;   // 检测到的边界框
    cv::Mat combinedMask;          // 组合分割掩码
    std::vector<cv::Mat> classMasks; // 按类别分组的掩码
};

// YOLO11分割推理类
class YOLO11SegInfer {
public:
    /**
     * @brief 构造函数
     * @param modelPath 模型文件路径
     * @param modelWidth 模型输入宽度
     * @param modelHeight 模型输入高度
     * @param confidenceThreshold 置信度阈值
     * @param nmsThreshold NMS阈值
     */
    YOLO11SegInfer(const std::string& modelPath, 
                   int modelWidth = 640, 
                   int modelHeight = 640,
                   float confidenceThreshold = 0.35f,
                   float nmsThreshold = 0.45f);
    
    ~YOLO11SegInfer();

    /**
     * @brief 初始化资源
     * @return 成功返回0，失败返回-1
     */
    int InitResource();

    /**
     * @brief 从NV12格式的ROS消息进行推理
     * @param msg NV12格式的图像消息
     * @param result 输出分割结果
     * @return 成功返回0，失败返回-1
     */
    int ProcessImageFromNV12(const sensor_msgs::msg::Image& msg, SegmentationResult& result);

    /**
     * @brief 设置置信度阈值
     * @param threshold 新的置信度阈值
     */
    void SetConfidenceThreshold(float threshold) { confidenceThreshold_ = threshold; }

    /**
     * @brief 设置NMS阈值
     * @param threshold 新的NMS阈值
     */
    void SetNMSThreshold(float threshold) { nmsThreshold_ = threshold; }

    /**
     * @brief 获取支持的类别名称列表
     * @return 类别名称向量
     */
    std::vector<std::string> GetClassNames() const { return classNames_; }

private:
    /**
     * @brief 释放资源
     */
    void ReleaseResource();

    /**
     * @brief 预处理NV12格式图像
     * @param msg NV12格式的图像消息
     * @return 成功返回0，失败返回-1
     */
    int PreprocessImageFromNV12(const sensor_msgs::msg::Image& msg);

    /**
     * @brief 执行推理
     * @param inferOutputs 推理输出结果
     * @return 成功返回0，失败返回-1
     */
    int DoInference(std::vector<InferenceOutput>& inferOutputs);

    /**
     * @brief 后处理推理结果
     * @param inferOutputs 推理输出
     * @param srcWidth 原始图像宽度
     * @param srcHeight 原始图像高度
     * @param result 输出分割结果
     * @return 成功返回0，失败返回-1
     */
    int PostprocessResult(const std::vector<InferenceOutput>& inferOutputs,
                          int srcWidth, int srcHeight,
                          SegmentationResult& result);

    /**
     * @brief 处理单个掩码
     * @param box 边界框信息
     * @param maskProtoData 掩码原型数据
     * @param srcWidth 原始图像宽度
     * @param srcHeight 原始图像高度
     * @return 生成的掩码图像
     */
    cv::Mat ProcessMask(const BoundBox& box, const float* maskProtoData, 
                        int srcWidth, int srcHeight);

    /**
     * @brief 非极大值抑制
     * @param boxes 输入边界框
     * @return 过滤后的边界框
     */
    std::vector<BoundBox> ApplyNMS(const std::vector<BoundBox>& boxes);

    /**
     * @brief 计算两个边界框的IoU
     * @param box1 边界框1
     * @param box2 边界框2
     * @return IoU值
     */
    float CalculateIoU(const BoundBox& box1, const BoundBox& box2);

private:
    // Ascend资源
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    
    // 模型参数
    std::string modelPath_;
    int modelWidth_;
    int modelHeight_;
    float confidenceThreshold_;
    float nmsThreshold_;
    
    // 推理数据
    ImageData resizedImage_;
    
    // 类别信息
    std::vector<std::string> classNames_;
    
    // 状态标志
    bool initialized_;
};

} // namespace environment_detection

#endif // YOLO11_SEG_INFER_HPP

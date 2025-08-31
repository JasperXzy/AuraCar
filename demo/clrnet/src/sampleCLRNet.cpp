#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

// 车道线点结构
struct LanePoint {
    float x;
    float y;
    LanePoint(float x_val, float y_val) : x(x_val), y(y_val) {}
};

// 车道线结构
struct Lane {
    float conf;
    float start_y;
    float start_x;
    float theta;
    float length;
    std::vector<float> offsets;
    
    Lane() : conf(0.0f), start_y(0.0f), start_x(0.0f), theta(0.0f), length(0.0f) {}
};

class SampleCLRNet
{
public:
    SampleCLRNet(const char *modelPath, int32_t S = 72, int32_t cutHeight = 270, 
                 int32_t imgW = 800, int32_t imgH = 320, float confThresh = 0.5, 
                 float nmsThres = 50.0f, int32_t nmsTopk = 5);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, string imagePath, size_t imageIndex, bool release);
    ~SampleCLRNet();

private:
    void ReleaseResource();
    std::vector<LanePoint> PostProcess(float* pred);
    std::vector<Lane> DecodeLanes(float* pred);
    std::vector<Lane> NMS(std::vector<Lane>& lanes);
    float LaneIoU(const Lane& laneA, const Lane& laneB);
    std::vector<LanePoint> DecodeLanePoints(const std::vector<Lane>& lanes);
    float Softmax(float* scores, int size);
    
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    const char *modelPath_;
    int32_t S_;
    int32_t cutHeight_;
    int32_t imgW_;
    int32_t imgH_;
    float confThresh_;
    float nmsThres_;
    int32_t nmsTopk_;
    int32_t nStrips_;
    int32_t nOffsets_;
    std::vector<float> anchorYs_;
    int32_t oriW_;
    int32_t oriH_;
};

SampleCLRNet::SampleCLRNet(const char *modelPath, int32_t S, int32_t cutHeight, 
                           int32_t imgW, int32_t imgH, float confThresh, 
                           float nmsThres, int32_t nmsTopk) 
    : modelPath_(modelPath), S_(S), cutHeight_(cutHeight), imgW_(imgW), imgH_(imgH),
      confThresh_(confThresh), nmsThres_(nmsThres), nmsTopk_(nmsTopk)
{
    nStrips_ = S_ - 1;
    nOffsets_ = S_;
    oriW_ = 1640;
    oriH_ = 590;
    
    // 初始化anchor_ys
    for (int i = 0; i < nOffsets_; ++i) {
        anchorYs_.push_back(1.0f - static_cast<float>(i) / nStrips_);
    }
}

SampleCLRNet::~SampleCLRNet()
{
    ReleaseResource();
}

Result SampleCLRNet::InitResource()
{
    // 初始化acl资源
    AclLiteError ret = aclResource_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // 从文件加载模型
    ret = model_.Init(modelPath_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleCLRNet::ProcessInput(string testImgPath)
{
    // 使用OpenCV进行图像预处理
    cv::Mat srcImage = cv::imread(testImgPath);
    if (srcImage.empty()) {
        ACLLITE_LOG_ERROR("Failed to read image: %s", testImgPath.c_str());
        return FAILED;
    }
    
    // 裁剪图像底部区域
    cv::Mat croppedImage = srcImage(cv::Rect(0, cutHeight_, srcImage.cols, srcImage.rows - cutHeight_));
    
    // 缩放到模型输入尺寸
    cv::Mat resizedImage;
    cv::resize(croppedImage, resizedImage, cv::Size(imgW_, imgH_));
    
    // 转换为RGB格式
    cv::Mat rgbImage;
    cv::cvtColor(resizedImage, rgbImage, cv::COLOR_BGR2RGB);
    
    // 归一化到[0,1]范围
    cv::Mat normalizedImage;
    rgbImage.convertTo(normalizedImage, CV_32F, 1.0/255.0);
    
    // 转换为NCHW格式 (1, 3, H, W)
    std::vector<cv::Mat> channels(3);
    cv::split(normalizedImage, channels);
    
    // 计算所需的内存大小
    size_t inputSize = 1 * 3 * imgH_ * imgW_ * sizeof(float);
    
    // 分配内存并复制数据
    std::shared_ptr<uint8_t> modelInputData(new uint8_t[inputSize]);
    float* inputData = reinterpret_cast<float*>(modelInputData.get());
    
    // 按NCHW格式排列数据
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < imgH_; ++h) {
            for (int w = 0; w < imgW_; ++w) {
                int srcIdx = h * imgW_ + w;
                int dstIdx = c * imgH_ * imgW_ + h * imgW_ + w;
                inputData[dstIdx] = channels[c].at<float>(srcIdx);
            }
        }
    }
    
    // 更新resizedImage_为模型输入数据
    resizedImage_.data = modelInputData;
    resizedImage_.size = inputSize;
    
    return SUCCESS;
}

Result SampleCLRNet::Inference(std::vector<InferenceOutput> &inferOutputs)
{
    // 创建模型输入数据集
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // 推理
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

float SampleCLRNet::Softmax(float* scores, int size)
{
    float maxVal = scores[0];
    for (int i = 1; i < size; ++i) {
        if (scores[i] > maxVal) {
            maxVal = scores[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        scores[i] = exp(scores[i] - maxVal);
        sum += scores[i];
    }
    
    for (int i = 0; i < size; ++i) {
        scores[i] /= sum;
    }
    
    return scores[1]; // 返回置信度分数
}

std::vector<Lane> SampleCLRNet::DecodeLanes(float* pred)
{
    std::vector<Lane> lanes;
    
    // pred shape: [1, 192, 78]
    int numLanes = 192;
    int laneDim = 78;
    
    for (int laneId = 0; laneId < numLanes; ++laneId) {
        float* laneData = pred + laneId * laneDim;
        
        // 计算置信度（不修改原始数据）
        float scores[2] = {laneData[0], laneData[1]};
        float conf = Softmax(scores, 2);
        
        if (conf > confThresh_) {
            Lane lane;
            lane.conf = conf;
            lane.start_y = laneData[2];
            lane.start_x = laneData[3];
            lane.theta = laneData[4];
            lane.length = laneData[5];
            
            // 提取offset数据
            for (int i = 0; i < nOffsets_; ++i) {
                lane.offsets.push_back(laneData[6 + i]);
            }
            
            lanes.push_back(lane);
        }
    }
    
    // 按置信度排序
    std::sort(lanes.begin(), lanes.end(), [](const Lane& a, const Lane& b) {
        return a.conf > b.conf;
    });
    
    return lanes;
}

float SampleCLRNet::LaneIoU(const Lane& laneA, const Lane& laneB)
{
    int startA = static_cast<int>(laneA.start_y * nStrips_ + 0.5f);
    int startB = static_cast<int>(laneB.start_y * nStrips_ + 0.5f);
    int start = std::max(startA, startB);
    
    int endA = startA + static_cast<int>(laneA.length * nStrips_ + 0.5f) - 1;
    int endB = startB + static_cast<int>(laneB.length * nStrips_ + 0.5f) - 1;
    int end = std::min(std::min(endA, endB), nStrips_);
    
    if (end < start) return 0.0f;
    
    float dist = 0.0f;
    int count = 0;
    for (int i = start; i <= end; ++i) {
        if (i < laneA.offsets.size() && i < laneB.offsets.size()) {
            dist += abs((laneA.offsets[i] - laneB.offsets[i]) * (imgW_ - 1));
            count++;
        }
    }
    
    return count > 0 ? dist / count : 0.0f;
}

std::vector<Lane> SampleCLRNet::NMS(std::vector<Lane>& lanes)
{
    std::vector<bool> removeFlags(lanes.size(), false);
    std::vector<Lane> keepLanes;
    
    for (size_t i = 0; i < lanes.size(); ++i) {
        if (removeFlags[i]) continue;
        
        keepLanes.push_back(lanes[i]);
        
        for (size_t j = i + 1; j < lanes.size(); ++j) {
            if (removeFlags[j]) continue;
            
            if (LaneIoU(lanes[i], lanes[j]) < nmsThres_) {
                removeFlags[j] = true;
            }
        }
    }
    
    return keepLanes;
}

std::vector<LanePoint> SampleCLRNet::DecodeLanePoints(const std::vector<Lane>& lanes)
{
    std::vector<LanePoint> allLanePoints;
    
    for (const auto& lane : lanes) {
        int start = static_cast<int>(lane.start_y * nStrips_ + 0.5f);
        int end = start + static_cast<int>(lane.length * nStrips_ + 0.5f) - 1;
        end = std::min(end, nStrips_);
        
        std::vector<LanePoint> points;
        for (int i = start; i <= end; ++i) {
            if (i < lane.offsets.size()) {
                float y = anchorYs_[i];
                float factor = static_cast<float>(cutHeight_) / oriH_;
                float ys = (1.0f - factor) * y + factor;
                points.emplace_back(lane.offsets[i], ys);
            }
        }
        
        allLanePoints.insert(allLanePoints.end(), points.begin(), points.end());
    }
    
    return allLanePoints;
}

std::vector<LanePoint> SampleCLRNet::PostProcess(float* pred)
{
    // 解码车道线
    std::vector<Lane> lanes = DecodeLanes(pred);
    
    // NMS处理
    std::vector<Lane> nmsLanes = NMS(lanes);
    
    // 限制输出数量
    if (nmsLanes.size() > nmsTopk_) {
        nmsLanes.resize(nmsTopk_);
    }
    
    // 解码为点坐标
    return DecodeLanePoints(nmsLanes);
}

Result SampleCLRNet::GetResult(std::vector<InferenceOutput> &inferOutputs,
                               string imagePath, size_t imageIndex, bool release)
{
    uint32_t outputDataBufId = 0;
    float *predBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());
    
    // 后处理
    std::vector<LanePoint> lanePoints = PostProcess(predBuff);
    
    // 读取源图像
    cv::Mat srcImage = cv::imread(imagePath);
    if (srcImage.empty()) {
        ACLLITE_LOG_ERROR("Failed to read source image: %s", imagePath.c_str());
        return FAILED;
    }
    
    int srcWidth = srcImage.cols;
    int srcHeight = srcImage.rows;
    
    // 在图像上绘制车道线点
    for (const auto& point : lanePoints) {
        int x = static_cast<int>(point.x * srcWidth);
        int y = static_cast<int>(point.y * srcHeight);
        
        // 确保坐标在图像范围内
        x = std::max(0, std::min(x, srcWidth - 1));
        y = std::max(0, std::min(y, srcHeight - 1));
        
        cv::circle(srcImage, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
    }
    
    // 保存结果图像
    string savePath = "clrnet_result_" + to_string(imageIndex) + ".jpg";
    cv::imwrite(savePath, srcImage);
    
    ACLLITE_LOG_INFO("CLRNet inference completed, detected %ld lane points", lanePoints.size());
    
    // 不需要手动释放内存
    return SUCCESS;
}

void SampleCLRNet::ReleaseResource()
{
    model_.DestroyResource();
    aclResource_.Release();
}

int main()
{
    const char *modelPath = "../model/clrnet.om";
    const string imagePath = "../data";
    const int32_t S = 72;
    const int32_t cutHeight = 270;
    const int32_t imgW = 800;
    const int32_t imgH = 320;
    const float confThresh = 0.3f;
    const float nmsThres = 50.0f;
    const int32_t nmsTopk = 5;

    // 检查图像目录
    DIR *dir = opendir(imagePath.c_str());
    if (dir == nullptr)
    {
        ACLLITE_LOG_ERROR("file folder does not exist, please create folder %s", imagePath.c_str());
        return FAILED;
    }
    
    vector<string> allPath;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0 || strcmp(entry->d_name, ".keep") == 0)
        {
            continue;
        }
        else
        {
            string name = entry->d_name;
            string imgDir = imagePath + "/" + name;
            allPath.push_back(imgDir);
        }
    }
    closedir(dir);

    if (allPath.size() == 0)
    {
        ACLLITE_LOG_ERROR("the directory is empty, please download image to %s", imagePath.c_str());
        return FAILED;
    }

    // 推理
    string fileName;
    bool release = false;
    SampleCLRNet sampleCLRNet(modelPath, S, cutHeight, imgW, imgH, confThresh, nmsThres, nmsTopk);
    Result ret = sampleCLRNet.InitResource();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }

    for (size_t i = 0; i < allPath.size(); i++)
    {
        if (i == allPath.size() - 1)
        {
            release = true;
        }
        std::vector<InferenceOutput> inferOutputs;
        fileName = allPath.at(i).c_str();
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        
        ret = sampleCLRNet.ProcessInput(fileName);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
            return FAILED;
        }
        
        ret = sampleCLRNet.Inference(inferOutputs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            return FAILED;
        }

        ret = sampleCLRNet.GetResult(inferOutputs, fileName, i, release);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
            return FAILED;
        }
        
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        ACLLITE_LOG_INFO("Inference elapsed time : %f s , fps is %f", elapsed.count(), 1 / elapsed.count());
    }
    return SUCCESS;
}

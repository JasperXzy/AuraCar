#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include "label.h"
#include <chrono>

using namespace std;
using namespace cv;
typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef struct BoundBox
{
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex;
    size_t index;
    vector<float> mask;
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2)
{
    return box1.score > box2.score;
}

class SampleYOLOV8_SEG
{
public:
    SampleYOLOV8_SEG(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, string imagePath, size_t imageIndex, bool release);
    ~SampleYOLOV8_SEG();

private:
    void ReleaseResource();
    cv::Mat ProcessMask(const BoundBox& box, const float* maskData, int srcWidth, int srcHeight);
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
};

SampleYOLOV8_SEG::SampleYOLOV8_SEG(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight) : modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight)
{
}

SampleYOLOV8_SEG::~SampleYOLOV8_SEG()
{
    ReleaseResource();
}

Result SampleYOLOV8_SEG::InitResource()
{
    // init acl resource
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

    // init dvpp resource
    ret = imageProcess_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    // load model from file
    ret = model_.Init(modelPath_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV8_SEG::ProcessInput(string testImgPath)
{
    // read image from file
    ImageData image;
    AclLiteError ret = ReadJpeg(image, testImgPath);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }

    // copy image from host to dvpp
    ImageData imageDevice;
    ret = CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    // image decoded from JPEG format to YUV
    ImageData yuvImage;
    ret = imageProcess_.JpegD(yuvImage, imageDevice);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }

    // zoom image to modelWidth_ * modelHeight_
    ret = imageProcess_.Resize(resizedImage_, yuvImage, modelWidth_, modelHeight_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV8_SEG::Inference(std::vector<InferenceOutput> &inferOutputs)
{
    // create input data set of model
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // inference
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

cv::Mat SampleYOLOV8_SEG::ProcessMask(const BoundBox& box, const float* maskProtoData, int srcWidth, int srcHeight)
{
    cv::Mat mask = cv::Mat::zeros(srcHeight, srcWidth, CV_8UC1);
    
    // if there is no mask proto data, use simplified ellipse mask
    if (maskProtoData == nullptr || box.mask.empty()) {
        // calculate the position of the detection box in the original image
        int x1 = max(0, (int)(box.x - box.width / 2));
        int y1 = max(0, (int)(box.y - box.height / 2));
        int x2 = min(srcWidth, (int)(box.x + box.width / 2));
        int y2 = min(srcHeight, (int)(box.y + box.height / 2));
        
        // create an ellipse mask
        cv::Point center((x1 + x2) / 2, (y1 + y2) / 2);
        cv::Size axes((x2 - x1) / 2, (y2 - y1) / 2);
        cv::ellipse(mask, center, axes, 0, 0, 360, cv::Scalar(255), -1);
        
        // use gaussian blur to make the mask more natural
        cv::GaussianBlur(mask, mask, cv::Size(9, 9), 0);
        return mask;
    }
    
    // use mask coefficients and mask proto to generate the real segmentation mask
    // the size of mask proto is usually 160x160
    int protoSize = 160;
    cv::Mat protoMask = cv::Mat::zeros(protoSize, protoSize, CV_32F);
    
    // linear combination of mask coefficients and mask proto
    for (int y = 0; y < protoSize; y++) {
        for (int x = 0; x < protoSize; x++) {
            float sum = 0.0f;
            for (size_t k = 0; k < box.mask.size() && k < 32; k++) {
                // mask proto data index: [k * protoSize * protoSize + y * protoSize + x]
                int protoIdx = k * protoSize * protoSize + y * protoSize + x;
                sum += box.mask[k] * maskProtoData[protoIdx];
            }
            protoMask.at<float>(y, x) = sum;
        }
    }
    
    // apply sigmoid activation function
    cv::Mat sigmoidMask;
    cv::exp(-protoMask, sigmoidMask);
    sigmoidMask = 1.0 / (1.0 + sigmoidMask);
    
    // resize the mask to the original image size
    cv::Mat resizedMask;
    cv::resize(sigmoidMask, resizedMask, cv::Size(srcWidth, srcHeight));
    
    // convert to 8-bit image
    cv::Mat mask8u;
    resizedMask.convertTo(mask8u, CV_8UC1, 255.0);
    
    // apply threshold
    cv::threshold(mask8u, mask, 128, 255, cv::THRESH_BINARY);
    
    // apply morphological operation to smooth the edges
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    return mask;
}

Result SampleYOLOV8_SEG::GetResult(std::vector<InferenceOutput> &inferOutputs,
                               string imagePath, size_t imageIndex, bool release)
{
    // YOLOv8 segmentation model output structure:
    // output0: [1, 116, 8400] - detection result (4 coordinates + 80 classes + 32 mask coefficients)
    // output1: [1, 32, 160, 160] - mask proto
    uint32_t outputDataBufId = 0;
    uint32_t maskProtoBufId = 1;
    
    float *classBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());
    float *maskProtoBuff = nullptr;
    
    // check if there is mask proto output
    if (inferOutputs.size() > 1) {
        maskProtoBuff = static_cast<float *>(inferOutputs[maskProtoBufId].data.get());
    }
    
    // confidence threshold
    float confidenceThreshold = 0.35;

    // class number
    size_t classNum = 80;

    // number of (x, y, width, hight)
    size_t offset = 4;

    // total number of boxs yolov8 seg [1,116,8400] (4坐标 + 80类别 + 32mask系数)
    size_t modelOutputBoxNum = 8400; 
    
    // mask prototype number
    size_t maskProtoNum = 32;

    // read source image from file
    cv::Mat srcImage = cv::imread(imagePath);
    int srcWidth = srcImage.cols;
    int srcHeight = srcImage.rows;

    // filter boxes by confidence threshold
    vector<BoundBox> boxes;
    size_t yIndex = 1;
    size_t widthIndex = 2;
    size_t heightIndex = 3;

    for (size_t i = 0; i < modelOutputBoxNum; ++i)
    {
        float maxValue = 0;
        size_t maxIndex = 0;
        for (size_t j = 0; j < classNum; ++j)
        {
            float value = classBuff[(offset + j) * modelOutputBoxNum + i];
            if (value > maxValue)
            {
                // index of class
                maxIndex = j;
                maxValue = value;
            }
        }

        if (maxValue > confidenceThreshold)
        {
            BoundBox box;
            box.x = classBuff[i] * srcWidth / modelWidth_;
            box.y = classBuff[yIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.width = classBuff[widthIndex * modelOutputBoxNum + i] * srcWidth / modelWidth_;
            box.height = classBuff[heightIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.score = maxValue;
            box.classIndex = maxIndex;
            box.index = i;
            
            // extract mask coefficients (from the 84th position, 32 coefficients in total)
            size_t maskCoeffStart = offset + classNum; // 4 + 80 = 84
            for (size_t k = 0; k < maskProtoNum; k++) {
                float coeff = classBuff[(maskCoeffStart + k) * modelOutputBoxNum + i];
                box.mask.push_back(coeff);
            }
            
            if (maxIndex < classNum)
            {
                boxes.push_back(box);
            }
        }
    }

    ACLLITE_LOG_INFO("filter boxes by confidence threshold > %f success, boxes size is %ld", confidenceThreshold, boxes.size());

    // filter boxes by NMS
    vector<BoundBox> result;
    result.clear();
    float NMSThreshold = 0.45;
    int32_t maxLength = modelWidth_ > modelHeight_ ? modelWidth_ : modelHeight_;
    std::sort(boxes.begin(), boxes.end(), sortScore);
    BoundBox boxMax;
    BoundBox boxCompare;
    while (boxes.size() != 0)
    {
        size_t index = 1;
        result.push_back(boxes[0]);
        while (boxes.size() > index)
        {
            boxMax.score = boxes[0].score;
            boxMax.classIndex = boxes[0].classIndex;
            boxMax.index = boxes[0].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxMax.x = boxes[0].x + maxLength * boxes[0].classIndex;
            boxMax.y = boxes[0].y + maxLength * boxes[0].classIndex;
            boxMax.width = boxes[0].width;
            boxMax.height = boxes[0].height;

            boxCompare.score = boxes[index].score;
            boxCompare.classIndex = boxes[index].classIndex;
            boxCompare.index = boxes[index].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxCompare.x = boxes[index].x + boxes[index].classIndex * maxLength;
            boxCompare.y = boxes[index].y + boxes[index].classIndex * maxLength;
            boxCompare.width = boxes[index].width;
            boxCompare.height = boxes[index].height;

            // the overlapping part of the two boxes
            float xLeft = max(boxMax.x, boxCompare.x);
            float yTop = max(boxMax.y, boxCompare.y);
            float xRight = min(boxMax.x + boxMax.width, boxCompare.x + boxCompare.width);
            float yBottom = min(boxMax.y + boxMax.height, boxCompare.y + boxCompare.height);
            float width = max(0.0f, xRight - xLeft);
            float hight = max(0.0f, yBottom - yTop);
            float area = width * hight;
            float iou = area / (boxMax.width * boxMax.height + boxCompare.width * boxCompare.height - area);

            // filter boxes by NMS threshold
            if (iou > NMSThreshold)
            {
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
        }
        boxes.erase(boxes.begin());
    }

    ACLLITE_LOG_INFO("filter boxes by NMS threshold > %f success, result size is %ld", NMSThreshold, result.size());
 
    // opencv draw label params
    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255); // BGR
    const vector<cv::Scalar> colors{
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)};

    int half = 2;
    cv::Mat maskOverlay = srcImage.clone();
    
    for (size_t i = 0; i < result.size(); ++i)
    {
        cv::Point leftUpPoint, rightBottomPoint;
        leftUpPoint.x = result[i].x - result[i].width / half;
        leftUpPoint.y = result[i].y - result[i].height / half;
        rightBottomPoint.x = result[i].x + result[i].width / half;
        rightBottomPoint.y = result[i].y + result[i].height / half;
        
        // draw bounding box
        cv::rectangle(srcImage, leftUpPoint, rightBottomPoint, colors[i % colors.size()], lineSolid);
        
        // process and draw mask
        cv::Mat mask = ProcessMask(result[i], maskProtoBuff, srcWidth, srcHeight);
        
        // draw semi-transparent color directly in the mask area
        cv::Scalar color = colors[i % colors.size()];
        cv::Mat maskRegion;
        maskOverlay.copyTo(maskRegion, mask);
        maskOverlay.setTo(color * 0.3, mask);
        
        string className = label[result[i].classIndex];
        string markString = to_string(result[i].score) + ":" + className;

        ACLLITE_LOG_INFO("object detect [%s] success", markString.c_str());

        cv::putText(srcImage, markString, cv::Point(leftUpPoint.x, leftUpPoint.y + labelOffset),
                    cv::FONT_HERSHEY_COMPLEX, fountScale, fountColor);
    }
    
    // save the result image
    string savePath = "out_" + to_string(imageIndex) + ".jpg";
    cv::imwrite(savePath, srcImage);
    
    // save the image with mask
    string maskSavePath = "mask_" + to_string(imageIndex) + ".jpg";
    cv::imwrite(maskSavePath, maskOverlay);
    
    if (release)
    {
        free(classBuff);
        classBuff = nullptr;
        if (maskProtoBuff != nullptr) {
            free(maskProtoBuff);
            maskProtoBuff = nullptr;
        }
    }
    return SUCCESS;
}

void SampleYOLOV8_SEG::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int main()
{
    const char *modelPath = "../model/yolov8n-seg.om";
    const string imagePath = "../data";
    const int32_t modelWidth = 640;
    const int32_t modelHeight = 640;

    // all images in dir
    DIR *dir = opendir(imagePath.c_str());
    if (dir == nullptr)
    {
        ACLLITE_LOG_ERROR("file folder does no exist, please create folder %s", imagePath.c_str());
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

    // inference
    string fileName;
    bool release = false;
    SampleYOLOV8_SEG sampleYOLO(modelPath, modelWidth, modelHeight);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }

    for (size_t i = 0; i < allPath.size(); i++)
    {
        if (allPath.size() == i)
        {
            release = true;
        }
        std::vector<InferenceOutput> inferOutputs;
        fileName = allPath.at(i).c_str();
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        ret = sampleYOLO.ProcessInput(fileName);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
            return FAILED;
        }
        
        ret = sampleYOLO.Inference(inferOutputs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            return FAILED;
        }

        ret = sampleYOLO.GetResult(inferOutputs, fileName, i, release);
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

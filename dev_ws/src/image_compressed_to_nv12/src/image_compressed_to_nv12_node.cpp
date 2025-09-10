#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>

// 华为昇腾CANN相关头文件
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"

/**
 * @brief 图像压缩格式转NV12格式节点，使用DVPP硬件加速
 * 
 * 订阅image_compressed话题，使用DVPP硬件解码为YUV420SP格式，
 * 然后发布为image_raw话题（nv12编码）
 */
class ImageCompressedToNV12Node : public rclcpp::Node
{
public:
    ImageCompressedToNV12Node() : Node("image_compressed_to_nv12_node")
    {
        // 声明参数
        this->declare_parameter("input_topic", "image_compressed");
        this->declare_parameter("output_topic", "image_raw");
        this->declare_parameter("frame_id", "camera_link");
        
        // 图像尺寸参数
        this->declare_parameter("image_width", 1920);
        this->declare_parameter("image_height", 1080);
        
        // DVPP硬件缩放参数
        this->declare_parameter("enable_hardware_resize", false);
        this->declare_parameter("resize_width", 1920);
        this->declare_parameter("resize_height", 1080);
        
        // 获取参数
        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        frame_id_ = this->get_parameter("frame_id").as_string();
        
        // 获取图像尺寸参数
        image_width_ = this->get_parameter("image_width").as_int();
        image_height_ = this->get_parameter("image_height").as_int();
        
        // 获取DVPP参数
        enable_hardware_resize_ = this->get_parameter("enable_hardware_resize").as_bool();
        resize_width_ = this->get_parameter("resize_width").as_int();
        resize_height_ = this->get_parameter("resize_height").as_int();
        
        RCLCPP_INFO(this->get_logger(), "Image Compressed to NV12 Node Starting");
        RCLCPP_INFO(this->get_logger(), "Input Topic: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Output Topic: %s", output_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Frame ID: %s", frame_id_.c_str());
        RCLCPP_INFO(this->get_logger(), "Image Size: %dx%d", image_width_, image_height_);
        RCLCPP_INFO(this->get_logger(), "DVPP Acceleration: enabled");
        RCLCPP_INFO(this->get_logger(), "Hardware Resize: %s", enable_hardware_resize_ ? "true" : "false");
        
        // 创建图像发布器
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(output_topic_, 10);
        
        // 创建压缩图像订阅器
        compressed_image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            input_topic_, 10,
            std::bind(&ImageCompressedToNV12Node::compressed_image_callback, this, std::placeholders::_1)
        );
        
        // 初始化资源
        if (!initialize_resources()) {
            RCLCPP_ERROR(this->get_logger(), "Resource initialization failed");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Node initialized successfully with DVPP acceleration");
        
        // 初始化帧计数器和时间
        frame_count_ = 0;
        start_time_ = this->now();
    }
    
    ~ImageCompressedToNV12Node()
    {
        cleanup_resources();
    }

private:
    /**
     * @brief 初始化华为昇腾CANN资源
     * @return 初始化成功返回true，失败返回false
     */
    bool initialize_resources()
    {
        // 初始化ACL资源
        AclLiteError ret = acl_resource_.Init();
        if (ret != ACLLITE_OK) {
            RCLCPP_ERROR(this->get_logger(), "ACL resource initialization failed");
            return false;
        }
        
        // 获取运行模式
        ret = aclrtGetRunMode(&run_mode_);
        if (ret != ACL_SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get run mode");
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Run mode: %s", (run_mode_ == ACL_HOST) ? "Host" : "Device");
        
        // 初始化DVPP图像处理
        ret = image_process_.Init();
        if (ret != ACLLITE_OK) {
            RCLCPP_ERROR(this->get_logger(), "DVPP image process initialization failed");
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "DVPP image process initialized successfully");
        
        return true;
    }
    
    /**
     * @brief 压缩图像回调函数
     * @param msg 压缩图像消息
     */
    void compressed_image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        try {
            RCLCPP_DEBUG(this->get_logger(), "Received compressed image (size: %zu bytes, format: %s)", 
                        msg->data.size(), msg->format.c_str());
            
            // 检查图像格式
            if (msg->format != "jpeg" && msg->format != "jpg") {
                RCLCPP_WARN(this->get_logger(), "Unsupported image format: %s, only JPEG is supported", msg->format.c_str());
                return;
            }
            
            // 使用DVPP硬件解码
            process_compressed_image_with_dvpp(msg);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in compressed_image_callback: %s", e.what());
        }
    }
    
    /**
     * @brief 使用DVPP硬件解码压缩图像为YUV420SP
     */
    void process_compressed_image_with_dvpp(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        try {
            RCLCPP_DEBUG(this->get_logger(), "Starting DVPP JPEG decode with AclLite (size: %zu bytes)", msg->data.size());
            
            // 创建输入JPEG图像数据结构
            ImageData jpeg_image;
            jpeg_image.format = PIXEL_FORMAT_U8C1;  // 使用U8C1格式表示JPEG数据
            jpeg_image.width = static_cast<uint32_t>(image_width_);
            jpeg_image.height = static_cast<uint32_t>(image_height_);
            jpeg_image.size = msg->data.size();
            jpeg_image.data = std::shared_ptr<uint8_t>(const_cast<uint8_t*>(msg->data.data()), [](uint8_t*) {});
            
            RCLCPP_DEBUG(this->get_logger(), "JPEG image dimensions: %dx%d", image_width_, image_height_);
            
            // 使用DVPP硬件解码JPEG
            ImageData yuv_image;
            AclLiteError ret = image_process_.JpegD(yuv_image, jpeg_image);
            if (ret != ACLLITE_OK) {
                RCLCPP_ERROR(this->get_logger(), "DVPP JPEG decode failed with error: %d", ret);
                return;
            }
            
            RCLCPP_DEBUG(this->get_logger(), "DVPP JPEG decode completed successfully (size: %ux%u)", 
                        yuv_image.width, yuv_image.height);
            
            // 使用DVPP硬件缩放
            const bool need_resize = enable_hardware_resize_ &&
                                     resize_width_ > 0 && resize_height_ > 0 &&
                                     (static_cast<int>(yuv_image.width) != resize_width_ ||
                                      static_cast<int>(yuv_image.height) != resize_height_);

            if (need_resize) {
                ImageData resized_yuv;
                RCLCPP_DEBUG(this->get_logger(), "DVPP resizing from %ux%u to %dx%d", 
                            yuv_image.width, yuv_image.height, resize_width_, resize_height_);
                ret = image_process_.Resize(resized_yuv, yuv_image, 
                                           static_cast<uint32_t>(resize_width_), 
                                           static_cast<uint32_t>(resize_height_));
                if (ret != ACLLITE_OK) {
                    RCLCPP_ERROR(this->get_logger(), "DVPP resize failed with error: %d", ret);
                    return;
                }

                if (resized_yuv.data && resized_yuv.size > 0) {
                    std::vector<uint8_t> yuv_data(resized_yuv.size);
                    memcpy(yuv_data.data(), resized_yuv.data.get(), resized_yuv.size);
                    publish_yuv420sp_image(yuv_data, resized_yuv.width, resized_yuv.height, msg->header);
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Invalid resized YUV image data from DVPP");
                }
            } else {
                // 直接发布YUV420SP格式的数据到image_raw
                if (yuv_image.data && yuv_image.size > 0) {
                    std::vector<uint8_t> yuv_data(yuv_image.size);
                    memcpy(yuv_data.data(), yuv_image.data.get(), yuv_image.size);
                    publish_yuv420sp_image(yuv_data, yuv_image.width, yuv_image.height, msg->header);
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Invalid YUV image data from DVPP");
                }
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in DVPP JPEG decode: %s", e.what());
        }
    }
    
    /**
     * @brief 发布YUV420SP格式的图像
     */
    void publish_yuv420sp_image(const std::vector<uint8_t>& yuv_data, uint32_t width, uint32_t height, 
                               const std_msgs::msg::Header& /* original_header */)
    {
        // 创建ROS图像消息，使用nv12编码
        auto image_msg = std::make_unique<sensor_msgs::msg::Image>();
        image_msg->header.stamp = this->now();
        image_msg->header.frame_id = frame_id_;
        image_msg->height = height;
        image_msg->width = width;
        image_msg->encoding = "nv12";
        image_msg->is_bigendian = false;
        image_msg->step = width;
        image_msg->data = yuv_data;
        
        // 发布图像
        image_pub_->publish(std::move(image_msg));
        
        // 更新帧计数器和显示信息
        frame_count_++;
        auto current_time = this->now();
        auto elapsed = (current_time - start_time_).seconds();
        
        if (frame_count_ % 30 == 0) {  // 每30帧显示一次统计信息
            double actual_fps = frame_count_ / elapsed;
            RCLCPP_INFO(this->get_logger(), "Published %d frames (%ux%u), Runtime: %.1f seconds, Actual FPS: %.1f fps", 
                       frame_count_, width, height, elapsed, actual_fps);
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Published YUV420SP image: %ux%u (frame #%d)", 
                     width, height, frame_count_);
    }
    
    /**
     * @brief 清理资源
     */
    void cleanup_resources()
    {
        // 清理DVPP资源
        image_process_.DestroyResource();
        
        // 清理ACL资源
        acl_resource_.Release();
    }
    
    // 基础参数
    std::string input_topic_;               // 输入话题名称
    std::string output_topic_;              // 输出话题名称
    std::string frame_id_;                  // 帧ID
    
    // 图像尺寸参数
    int image_width_;                       // 输入图像宽度
    int image_height_;                      // 输入图像高度
    
    // DVPP优化参数
    bool enable_hardware_resize_;           // 是否启用硬件缩放
    int resize_width_;                      // 缩放宽度
    int resize_height_;                     // 缩放高度
    
    // ROS2相关
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;                           // 图像发布器
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_image_sub_;   // 压缩图像订阅器
    
    // 帧计数器和时间统计
    int frame_count_;                       // 已发布的总帧数
    rclcpp::Time start_time_;               // FPS计算的开始时间
    
    // 华为昇腾CANN相关
    AclLiteResource acl_resource_;          // ACL资源管理器
    AclLiteImageProc image_process_;        // DVPP图像处理器
    aclrtRunMode run_mode_;                 // 运行模式
};

/**
 * @brief 主函数
 * @param argc 命令行参数数量
 * @param argv 命令行参数
 * @return 退出码
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<ImageCompressedToNV12Node>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}

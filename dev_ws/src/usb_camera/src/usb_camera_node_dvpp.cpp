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

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
}

/**
 * @brief 使用华为昇腾DVPP硬件加速的USB摄像头节点
 * 
 * 支持硬件解码、硬件缩放、硬件格式转换
 */
class USBCameraNodeDVPP : public rclcpp::Node
{
public:
    USBCameraNodeDVPP() : Node("usb_camera_node_dvpp")
    {
        // 声明参数
        this->declare_parameter("device_path", "/dev/video0");
        this->declare_parameter("width", 1920);
        this->declare_parameter("height", 1080);
        this->declare_parameter("fps", 30);
        this->declare_parameter("publish_compressed", true);
        
        // DVPP硬件缩放参数
        this->declare_parameter("enable_hardware_resize", false);
        this->declare_parameter("resize_width", 1920);
        this->declare_parameter("resize_height", 1080);
        
        // 获取参数
        device_path_ = this->get_parameter("device_path").as_string();
        width_ = this->get_parameter("width").as_int();
        height_ = this->get_parameter("height").as_int();
        fps_ = this->get_parameter("fps").as_int();
        publish_compressed_ = this->get_parameter("publish_compressed").as_bool();
        
        // 获取DVPP参数
        enable_hardware_resize_ = this->get_parameter("enable_hardware_resize").as_bool();
        resize_width_ = this->get_parameter("resize_width").as_int();
        resize_height_ = this->get_parameter("resize_height").as_int();
        
        RCLCPP_INFO(this->get_logger(), "USB Camera Node DVPP Starting");
        RCLCPP_INFO(this->get_logger(), "Device Path: %s", device_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Resolution: %dx%d", width_, height_);
        RCLCPP_INFO(this->get_logger(), "Frame Rate: %d fps", fps_);
        RCLCPP_INFO(this->get_logger(), "Pixel Format: MJPEG");
        RCLCPP_INFO(this->get_logger(), "Publish Compressed: %s", publish_compressed_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "DVPP Acceleration: enabled");
        RCLCPP_INFO(this->get_logger(), "Hardware Resize: %s", enable_hardware_resize_ ? "true" : "false");
        
        // 创建图像发布器
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", 10);
        
        // 根据配置决定是否创建压缩图像发布器
        if (publish_compressed_) {
            compressed_image_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("camera/image_compressed", 10);
            RCLCPP_INFO(this->get_logger(), "Compressed image publisher created for topic: camera/image_compressed");
        } else {
            RCLCPP_INFO(this->get_logger(), "Compressed image publishing disabled");
        }
        
        // 初始化资源
        if (!initialize_resources()) {
            RCLCPP_ERROR(this->get_logger(), "Resource initialization failed");
            return;
        }
        
        // 创建定时器，控制发布频率
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / fps_),
            std::bind(&USBCameraNodeDVPP::capture_and_publish, this)
        );
        
        RCLCPP_INFO(this->get_logger(), "Camera initialized successfully with DVPP acceleration");
        
        // 初始化帧计数器和时间
        frame_count_ = 0;
        start_time_ = this->now();
    }
    
    ~USBCameraNodeDVPP()
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
        
        // 初始化FFmpeg（用于摄像头读取）
        if (!initialize_ffmpeg()) {
            RCLCPP_ERROR(this->get_logger(), "FFmpeg initialization failed");
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief 初始化FFmpeg摄像头
     * @return 初始化成功返回true，失败返回false
     */
    bool initialize_ffmpeg()
    {
        // 注册FFmpeg设备
        avdevice_register_all();
        
        // 创建格式上下文
        format_context_ = avformat_alloc_context();
        if (!format_context_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to allocate format context");
            return false;
        }
        
        // 设置输入格式
        input_format_ = av_find_input_format("video4linux2");
        if (!input_format_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to find video4linux2 input format");
            return false;
        }
        
        // 设置选项
        AVDictionary* options = nullptr;
        std::string video_size = std::to_string(width_) + "x" + std::to_string(height_);
        std::string framerate = std::to_string(fps_);
        
        av_dict_set(&options, "video_size", video_size.c_str(), 0);
        av_dict_set(&options, "framerate", framerate.c_str(), 0);
        av_dict_set(&options, "pixel_format", "mjpeg", 0);
        
        // 打开输入流
        int ret = avformat_open_input(&format_context_, device_path_.c_str(), input_format_, &options);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
            RCLCPP_ERROR(this->get_logger(), "Failed to open camera: %s", errbuf);
            av_dict_free(&options);
            return false;
        }
        
        av_dict_free(&options);
        
        // 查找流信息
        if (avformat_find_stream_info(format_context_, nullptr) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to find stream information");
            return false;
        }
        
        // 查找视频流
        video_stream_index_ = -1;
        for (unsigned int i = 0; i < format_context_->nb_streams; i++) {
            if (format_context_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index_ = i;
                break;
            }
        }
        
        if (video_stream_index_ == -1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to find video stream");
            return false;
        }
        
        // 获取编解码器参数
        codec_params_ = format_context_->streams[video_stream_index_]->codecpar;
        
        // 查找解码器
        codec_ = avcodec_find_decoder(codec_params_->codec_id);
        if (!codec_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to find decoder");
            return false;
        }
        
        // 创建解码器上下文
        codec_context_ = avcodec_alloc_context3(codec_);
        if (!codec_context_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to allocate decoder context");
            return false;
        }
        
        // 将参数复制到解码器上下文
        if (avcodec_parameters_to_context(codec_context_, codec_params_) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to copy codec parameters");
            return false;
        }
        
        // 打开解码器
        if (avcodec_open2(codec_context_, codec_, nullptr) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open decoder");
            return false;
        }
        
        // 分配帧和数据包
        frame_ = av_frame_alloc();
        packet_ = av_packet_alloc();
        
        if (!frame_ || !packet_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to allocate frame or packet");
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "FFmpeg initialized successfully");
        RCLCPP_INFO(this->get_logger(), "Codec: %s", codec_->name);
        RCLCPP_INFO(this->get_logger(), "Format: %d", codec_params_->format);
        
        return true;
    }
    
    /**
     * @brief 从摄像头捕获帧并发布为ROS2消息
     */
    void capture_and_publish()
    {
        if (!format_context_ || !packet_) {
            RCLCPP_ERROR(this->get_logger(), "Format context or packet is null");
            return;
        }
        
        try {
            // 读取帧
            int ret = av_read_frame(format_context_, packet_);
            if (ret >= 0) {
                if (packet_->stream_index == video_stream_index_) {
                    // 直接处理MJPEG数据包
                    process_mjpeg_packet();
                }
                av_packet_unref(packet_);
            } else if (ret != AVERROR(EAGAIN)) {
                RCLCPP_WARN(this->get_logger(), "Failed to read frame: %d", ret);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in capture_and_publish: %s", e.what());
        }
    }
    
    /**
     * @brief 处理MJPEG数据包
     */
    void process_mjpeg_packet()
    {
        if (!packet_ || packet_->size <= 0) {
            RCLCPP_WARN(this->get_logger(), "No packet data available");
            return;
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Processing MJPEG packet (size: %d bytes)", packet_->size);
        
        // 获取原始MJPEG数据
        std::vector<uint8_t> jpeg_data(packet_->data, packet_->data + packet_->size);
        
        // 首先发布压缩图像（如果启用）
        if (publish_compressed_) {
            auto compressed_msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
            compressed_msg->header.stamp = this->now();
            compressed_msg->header.frame_id = "camera_link";
            compressed_msg->format = "jpeg";
            compressed_msg->data = jpeg_data;
            compressed_image_pub_->publish(*compressed_msg);
            
            RCLCPP_DEBUG(this->get_logger(), "Published compressed MJPEG image (size: %zu bytes)", jpeg_data.size());
        }
        
        // 使用DVPP硬件解码（固定启用）
        RCLCPP_DEBUG(this->get_logger(), "Using DVPP hardware decode");
        process_mjpeg_with_dvpp(jpeg_data);
    }
    
    /**
     * @brief 使用DVPP硬件解码MJPEG为YUV420SP
     */
    void process_mjpeg_with_dvpp(const std::vector<uint8_t>& jpeg_data)
    {
        try {
            RCLCPP_DEBUG(this->get_logger(), "Starting DVPP JPEG decode with AclLite (size: %zu bytes)", jpeg_data.size());
            
            // 创建输入JPEG图像数据结构
            ImageData jpeg_image;
            jpeg_image.format = PIXEL_FORMAT_U8C1;  // 使用U8C1格式表示JPEG数据
            jpeg_image.width = width_;
            jpeg_image.height = height_;
            jpeg_image.size = jpeg_data.size();
            jpeg_image.data = std::shared_ptr<uint8_t>(const_cast<uint8_t*>(jpeg_data.data()), [](uint8_t*) {});
            
            // 使用DVPP硬件解码JPEG
            ImageData yuv_image;
            AclLiteError ret = image_process_.JpegD(yuv_image, jpeg_image);
            if (ret != ACLLITE_OK) {
                RCLCPP_ERROR(this->get_logger(), "DVPP JPEG decode failed with error: %d", ret);
                return;
            }
            
            RCLCPP_DEBUG(this->get_logger(), "DVPP JPEG decode completed successfully");
            
            // 使用DVPP硬件缩放
            const bool need_resize = enable_hardware_resize_ &&
                                     resize_width_ > 0 && resize_height_ > 0 &&
                                     (static_cast<int>(yuv_image.width) != resize_width_ ||
                                      static_cast<int>(yuv_image.height) != resize_height_);

            if (need_resize) {
                ImageData resized_yuv;
                RCLCPP_DEBUG(this->get_logger(), "DVPP resizing from %ux%u to %dx%d", yuv_image.width, yuv_image.height, resize_width_, resize_height_);
                ret = image_process_.Resize(resized_yuv, yuv_image, static_cast<uint32_t>(resize_width_), static_cast<uint32_t>(resize_height_));
                if (ret != ACLLITE_OK) {
                    RCLCPP_ERROR(this->get_logger(), "DVPP resize failed with error: %d", ret);
                    return;
                }

                if (resized_yuv.data && resized_yuv.size > 0) {
                    std::vector<uint8_t> yuv_data(resized_yuv.size);
                    memcpy(yuv_data.data(), resized_yuv.data.get(), resized_yuv.size);
                    publish_yuv420sp_image(yuv_data, resized_yuv.width, resized_yuv.height);
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Invalid resized YUV image data from DVPP");
                }
            } else {
                // 直接发布YUV420SP格式的数据到image_raw
                if (yuv_image.data && yuv_image.size > 0) {
                    std::vector<uint8_t> yuv_data(yuv_image.size);
                    memcpy(yuv_data.data(), yuv_image.data.get(), yuv_image.size);
                    publish_yuv420sp_image(yuv_data, yuv_image.width, yuv_image.height);
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
    void publish_yuv420sp_image(const std::vector<uint8_t>& yuv_data, uint32_t width, uint32_t height)
    {
        // 创建OpenCV Mat包装YUV420SP数据
        cv::Mat nv12_frame(height * 3 / 2, width, CV_8UC1, (void*)yuv_data.data());
        
        // 创建ROS图像消息，使用nv12编码
        auto image_msg = std::make_unique<sensor_msgs::msg::Image>();
        image_msg->header.stamp = this->now();
        image_msg->header.frame_id = "camera_link";
        image_msg->height = height;
        image_msg->width = width;
        image_msg->encoding = "nv12";
        image_msg->is_bigendian = false;
        image_msg->step = width;
        image_msg->data.assign(nv12_frame.data, nv12_frame.data + nv12_frame.total() * nv12_frame.elemSize());
        
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
        // 清理FFmpeg资源
        if (packet_) {
            av_packet_free(&packet_);
        }
        if (frame_) {
            av_frame_free(&frame_);
        }
        if (codec_context_) {
            avcodec_close(codec_context_);
            avcodec_free_context(&codec_context_);
        }
        if (format_context_) {
            avformat_close_input(&format_context_);
            avformat_free_context(format_context_);
        }
        
    // 清理DVPP资源（始终销毁）
    image_process_.DestroyResource();
        
        // 清理ACL资源
        acl_resource_.Release();
    }
    
    // 基础参数
    std::string device_path_;           // 摄像头设备路径
    int width_;                         // 图像宽度
    int height_;                        // 图像高度
    int fps_;                           // 帧率
    bool publish_compressed_;           // 是否发布压缩图像
    
    // DVPP优化参数
    bool enable_hardware_resize_;       // 是否启用硬件缩放
    int resize_width_;                  // 缩放宽度
    int resize_height_;                 // 缩放高度
    
    // ROS2相关
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;                       // 图像发布器
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_image_pub_;  // 压缩图像发布器
    rclcpp::TimerBase::SharedPtr timer_;                                                    // 帧捕获定时器
    
    // 帧计数器和时间统计
    int frame_count_;                   // 已发布的总帧数
    rclcpp::Time start_time_;           // FPS计算的开始时间
    
    // 华为昇腾CANN相关
    AclLiteResource acl_resource_;      // ACL资源管理器
    AclLiteImageProc image_process_;    // DVPP图像处理器
    aclrtRunMode run_mode_;             // 运行模式
    
    // FFmpeg相关变量
    AVFormatContext* format_context_;   // 输入格式上下文
    AVInputFormat* input_format_;       // 输入格式 (video4linux2)
    AVCodecParameters* codec_params_;   // 编解码器参数
    AVCodec* codec_;                    // 编解码器
    AVCodecContext* codec_context_;     // 编解码器上下文
    AVFrame* frame_;                    // 解码后的帧
    AVPacket* packet_;                  // 读取用的数据包
    int video_stream_index_;            // 视频流索引
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
    
    auto node = std::make_shared<USBCameraNodeDVPP>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}

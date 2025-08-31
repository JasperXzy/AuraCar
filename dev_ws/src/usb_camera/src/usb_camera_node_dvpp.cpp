#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
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
        this->declare_parameter("pixel_format", "mjpeg");
        
        // DVPP优化参数
        this->declare_parameter("enable_dvpp", true);
        this->declare_parameter("dvpp_output_format", "bgr888");
        this->declare_parameter("enable_hardware_resize", false);
        this->declare_parameter("resize_width", 1920);
        this->declare_parameter("resize_height", 1080);
        
        // 获取参数
        device_path_ = this->get_parameter("device_path").as_string();
        width_ = this->get_parameter("width").as_int();
        height_ = this->get_parameter("height").as_int();
        fps_ = this->get_parameter("fps").as_int();
        pixel_format_ = this->get_parameter("pixel_format").as_string();
        
        // 获取DVPP参数
        enable_dvpp_ = this->get_parameter("enable_dvpp").as_bool();
        dvpp_output_format_ = this->get_parameter("dvpp_output_format").as_string();
        enable_hardware_resize_ = this->get_parameter("enable_hardware_resize").as_bool();
        resize_width_ = this->get_parameter("resize_width").as_int();
        resize_height_ = this->get_parameter("resize_height").as_int();
        
        RCLCPP_INFO(this->get_logger(), "USB Camera Node DVPP Starting");
        RCLCPP_INFO(this->get_logger(), "Device Path: %s", device_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Resolution: %dx%d", width_, height_);
        RCLCPP_INFO(this->get_logger(), "Frame Rate: %d fps", fps_);
        RCLCPP_INFO(this->get_logger(), "Pixel Format: %s", pixel_format_.c_str());
        RCLCPP_INFO(this->get_logger(), "Enable DVPP: %s", enable_dvpp_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "DVPP Output Format: %s", dvpp_output_format_.c_str());
        RCLCPP_INFO(this->get_logger(), "Hardware Resize: %s", enable_hardware_resize_ ? "true" : "false");
        
        // 创建图像发布器
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", 10);
        
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
        if (enable_dvpp_) {
            ret = image_process_.Init();
            if (ret != ACLLITE_OK) {
                RCLCPP_ERROR(this->get_logger(), "DVPP image process initialization failed");
                return false;
            }
            RCLCPP_INFO(this->get_logger(), "DVPP image process initialized successfully");
        }
        
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
        
        // 设置像素格式
        if (pixel_format_ == "yuyv") {
            av_dict_set(&options, "pixel_format", "yuyv422", 0);
        } else {
            av_dict_set(&options, "pixel_format", "mjpeg", 0);
        }
        
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
        if (!format_context_ || !codec_context_) {
            return;
        }
        
        // 读取帧
        if (av_read_frame(format_context_, packet_) >= 0) {
            if (packet_->stream_index == video_stream_index_) {
                // 发送数据包到解码器
                int ret = avcodec_send_packet(codec_context_, packet_);
                if (ret < 0) {
                    RCLCPP_WARN(this->get_logger(), "Failed to send packet to decoder");
                    av_packet_unref(packet_);
                    return;
                }
                
                // 接收解码后的帧
                ret = avcodec_receive_frame(codec_context_, frame_);
                if (ret == 0) {
                    // 使用DVPP处理图像并发布
                    process_frame_with_dvpp();
                }
            }
            av_packet_unref(packet_);
        }
    }
    
    /**
     * @brief 使用DVPP处理帧数据
     */
    void process_frame_with_dvpp()
    {
        if (!frame_ || !frame_->data[0]) {
            return;
        }
        
        cv::Mat cv_frame;
        
        // 检查FFmpeg解码后的格式
        int frame_format = frame_->format;
        // RCLCPP_INFO(this->get_logger(), "Frame format: %d (AV_PIX_FMT_YUV420P=%d), pixel_format: %s", 
        //            frame_format, AV_PIX_FMT_YUV420P, pixel_format_.c_str());
        
        if (!enable_dvpp_) {
            RCLCPP_ERROR(this->get_logger(), "DVPP is disabled, cannot process frames");
            return;
        }
        
        // 使用DVPP硬件加速处理
        if (pixel_format_ == "mjpeg" && frame_format == 13) {  // 13是YUV420P的实际值
            // FFmpeg已经将MJPG解码为YUV420P，直接处理YUV数据
            // RCLCPP_INFO(this->get_logger(), "Processing YUV420P with DVPP");
            process_yuv420p_with_dvpp(cv_frame);
        } else if (pixel_format_ == "mjpeg" && frame_format != 13) {
            // 真正的JPEG数据，使用DVPP解码
            // RCLCPP_INFO(this->get_logger(), "Processing JPEG with DVPP");
            process_mjpg_with_dvpp(cv_frame);
        } else if (pixel_format_ == "yuyv") {
            // 处理YUYV格式
            // RCLCPP_INFO(this->get_logger(), "Processing YUYV with DVPP");
            process_yuyv_with_dvpp(cv_frame);
        } else {
            // 其他格式，DVPP不支持，直接报错
            RCLCPP_ERROR(this->get_logger(), "Unsupported pixel format: %s with frame format: %d", 
                       pixel_format_.c_str(), frame_format);
            return;
        }
        
        if (cv_frame.empty()) {
            RCLCPP_ERROR(this->get_logger(), "DVPP image processing failed");
            return;
        }
        
        // 发布图像
        publish_image(cv_frame);
    }
    
    /**
     * @brief 使用DVPP处理MJPG格式图像
     */
    void process_mjpg_with_dvpp(cv::Mat& cv_frame)
    {
        RCLCPP_DEBUG(this->get_logger(), "Processing MJPG with DVPP");
        
        // 创建JPEG数据
        ImageData jpeg_image;
        jpeg_image.format = PIXEL_FORMAT_U8C1;  // 使用U8C1格式表示JPEG数据
        jpeg_image.width = width_;
        jpeg_image.height = height_;
        jpeg_image.size = frame_->linesize[0];
        jpeg_image.data = std::shared_ptr<uint8_t>(frame_->data[0], [](uint8_t*) {});  // 移除未使用的参数警告
        
        // 使用DVPP硬件解码JPEG
        ImageData yuv_image;
        AclLiteError ret = image_process_.JpegD(yuv_image, jpeg_image);
        if (ret != ACLLITE_OK) {
            RCLCPP_ERROR(this->get_logger(), "DVPP JPEG decode failed with error: %d", ret);
            return;
        }
        
        // 如果需要硬件缩放
        if (enable_hardware_resize_) {
            ImageData resized_image;
            ret = image_process_.Resize(resized_image, yuv_image, resize_width_, resize_height_);
            if (ret != ACLLITE_OK) {
                RCLCPP_ERROR(this->get_logger(), "DVPP resize failed with error: %d", ret);
                return;
            }
            yuv_image = resized_image;
        }
        
        // 转换为OpenCV格式
        convert_yuv_to_opencv(yuv_image, cv_frame);
    }
    
    /**
     * @brief 使用DVPP处理YUYV格式图像
     */
    void process_yuyv_with_dvpp(cv::Mat& cv_frame)
    {
        RCLCPP_DEBUG(this->get_logger(), "Processing YUYV with DVPP");
        
        // 创建YUYV数据
        ImageData yuyv_image;
        yuyv_image.format = PIXEL_FORMAT_YUV_SEMIPLANAR_422;
        yuyv_image.width = width_;
        yuyv_image.height = height_;
        yuyv_image.size = frame_->linesize[0] * height_;
        yuyv_image.data = std::shared_ptr<uint8_t>(frame_->data[0], [](uint8_t*) {});  // 移除未使用的参数警告
        
        // 如果需要硬件缩放
        if (enable_hardware_resize_) {
            ImageData resized_image;
            AclLiteError ret = image_process_.Resize(resized_image, yuyv_image, resize_width_, resize_height_);
            if (ret != ACLLITE_OK) {
                RCLCPP_ERROR(this->get_logger(), "DVPP resize failed with error: %d", ret);
                return;
            }
            yuyv_image = resized_image;
        }
        
        // 转换为OpenCV格式
        convert_yuv_to_opencv(yuyv_image, cv_frame);
    }
    
    /**
     * @brief 使用DVPP处理YUV420P格式图像（FFmpeg已解码）
     */
    void process_yuv420p_with_dvpp(cv::Mat& cv_frame)
    {
        // RCLCPP_INFO(this->get_logger(), "Processing YUV420P with DVPP - Start");
        
        // 创建YUV420P数据
        ImageData yuv_image;
        yuv_image.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
        yuv_image.width = width_;
        yuv_image.height = height_;
        yuv_image.size = width_ * height_ * 3 / 2;  // YUV420P大小
        
        // RCLCPP_INFO(this->get_logger(), "YUV420P size: %d, width: %d, height: %d", 
        //            yuv_image.size, yuv_image.width, yuv_image.height);
        
        // 分配内存并复制YUV数据
        std::shared_ptr<uint8_t> yuv_data(new uint8_t[yuv_image.size], [](uint8_t* p) { delete[] p; });
        
        // 复制Y平面
        int y_size = width_ * height_;
        memcpy(yuv_data.get(), frame_->data[0], y_size);
        // RCLCPP_INFO(this->get_logger(), "Copied Y plane: %d bytes", y_size);
        
        // 复制U平面
        int u_size = width_ * height_ / 4;
        memcpy(yuv_data.get() + y_size, frame_->data[1], u_size);
        // RCLCPP_INFO(this->get_logger(), "Copied U plane: %d bytes", u_size);
        
        // 复制V平面
        memcpy(yuv_data.get() + y_size + u_size, frame_->data[2], u_size);
        // RCLCPP_INFO(this->get_logger(), "Copied V plane: %d bytes", u_size);
        
        yuv_image.data = yuv_data;
        
        // 如果需要硬件缩放
        if (enable_hardware_resize_) {
            // RCLCPP_INFO(this->get_logger(), "Applying hardware resize: %dx%d -> %dx%d", 
            //            width_, height_, resize_width_, resize_height_);
            ImageData resized_image;
            AclLiteError ret = image_process_.Resize(resized_image, yuv_image, resize_width_, resize_height_);
            if (ret != ACLLITE_OK) {
                RCLCPP_ERROR(this->get_logger(), "DVPP resize failed with error: %d", ret);
                return;
            }
            yuv_image = resized_image;
            // RCLCPP_INFO(this->get_logger(), "Hardware resize completed");
        }
        
        // 转换为OpenCV格式
        // RCLCPP_INFO(this->get_logger(), "Converting YUV to OpenCV format");
        convert_yuv_to_opencv(yuv_image, cv_frame);
        
        if (cv_frame.empty()) {
            RCLCPP_ERROR(this->get_logger(), "convert_yuv_to_opencv failed - cv_frame is empty");
        } else {
            // RCLCPP_INFO(this->get_logger(), "convert_yuv_to_opencv succeeded - cv_frame size: %dx%d", 
            //            cv_frame.cols, cv_frame.rows);
        }
    }
    
    /**
     * @brief 将YUV图像转换为OpenCV格式
     */
    void convert_yuv_to_opencv(const ImageData& yuv_image, cv::Mat& cv_frame)
    {
        // RCLCPP_INFO(this->get_logger(), "convert_yuv_to_opencv - Start");
        // RCLCPP_INFO(this->get_logger(), "Input YUV image: width=%d, height=%d, size=%d, format=%d", 
        //            yuv_image.width, yuv_image.height, yuv_image.size, yuv_image.format);
        
        if (!yuv_image.data) {
            RCLCPP_ERROR(this->get_logger(), "convert_yuv_to_opencv failed - yuv_image.data is null");
            return;
        }
        
        // RCLCPP_INFO(this->get_logger(), "DVPP output format: %s", dvpp_output_format_.c_str());
        
        // 根据输出格式要求转换
        if (dvpp_output_format_ == "bgr888") {
            // RCLCPP_INFO(this->get_logger(), "Creating BGR image: %dx%d", yuv_image.width, yuv_image.height);
            
            // 创建BGR图像
            cv_frame = cv::Mat(yuv_image.height, yuv_image.width, CV_8UC3);
            
            // 这里需要根据具体的YUV格式进行转换
            // 由于DVPP输出通常是YUV420SP格式，需要转换为BGR
            cv::Mat yuv_mat(yuv_image.height * 3 / 2, yuv_image.width, CV_8UC1, 
                           const_cast<uint8_t*>(yuv_image.data.get()));
            
            // RCLCPP_INFO(this->get_logger(), "YUV mat created: %dx%d, total size: %d", 
            //            yuv_mat.rows, yuv_mat.cols, yuv_mat.total() * yuv_mat.elemSize());
            
            try {
                cv::cvtColor(yuv_mat, cv_frame, cv::COLOR_YUV2BGR_NV12);
                // RCLCPP_INFO(this->get_logger(), "cv::cvtColor succeeded - BGR frame: %dx%d", 
                //            cv_frame.cols, cv_frame.rows);
            } catch (const cv::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv::cvtColor failed with exception: %s", e.what());
                cv_frame = cv::Mat();  // 清空帧
            }

        } else {
            // RCLCPP_INFO(this->get_logger(), "Using direct copy for format: %s", dvpp_output_format_.c_str());
            
            // 其他格式
            cv_frame = cv::Mat(yuv_image.height, yuv_image.width, CV_8UC3);
            memcpy(cv_frame.data, yuv_image.data.get(), yuv_image.size);
            // RCLCPP_INFO(this->get_logger(), "Direct copy completed - frame: %dx%d", 
            //            cv_frame.cols, cv_frame.rows);
        }
        
        if (cv_frame.empty()) {
            RCLCPP_ERROR(this->get_logger(), "convert_yuv_to_opencv failed - final cv_frame is empty");
        } else {
            // RCLCPP_INFO(this->get_logger(), "convert_yuv_to_opencv succeeded - final cv_frame: %dx%d, type: %d", 
            //            cv_frame.cols, cv_frame.rows, cv_frame.type());
        }
    }
    
    /**
     * @brief 发布图像消息
     */
    void publish_image(const cv::Mat& cv_frame)
    {
        // 创建ROS图像消息
        auto ros_image = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_frame);
        ros_image.header.stamp = this->now();
        ros_image.header.frame_id = "camera_link";
        
        // 发布图像
        image_pub_->publish(*ros_image.toImageMsg());
        
        // 更新帧计数器和显示信息
        frame_count_++;
        auto current_time = this->now();
        auto elapsed = (current_time - start_time_).seconds();
        
        if (frame_count_ % 30 == 0) {  // 每30帧显示一次统计信息
            double actual_fps = frame_count_ / elapsed;
            RCLCPP_INFO(this->get_logger(), "Published %d frames, Runtime: %.1f seconds, Actual FPS: %.1f fps, Image size: %dx%d", 
                       frame_count_, elapsed, actual_fps, cv_frame.cols, cv_frame.rows);
        }
        
        // RCLCPP_DEBUG(this->get_logger(), "Published image: %dx%d (frame #%d)", 
        //              cv_frame.cols, cv_frame.rows, frame_count_);
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
        
        // 清理DVPP资源
        if (enable_dvpp_) {
            image_process_.DestroyResource();
        }
        
        // 清理ACL资源
        acl_resource_.Release();
    }
    
    // 基础参数
    std::string device_path_;           // 摄像头设备路径
    int width_;                         // 图像宽度
    int height_;                        // 图像高度
    int fps_;                           // 帧率
    std::string pixel_format_;          // 像素格式 (mjpeg 或 yuyv)
    
    // DVPP优化参数
    bool enable_dvpp_;                  // 是否启用DVPP
    std::string dvpp_output_format_;    // DVPP输出格式
    bool enable_hardware_resize_;       // 是否启用硬件缩放
    int resize_width_;                  // 缩放宽度
    int resize_height_;                 // 缩放高度
    
    // ROS2相关
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;  // 图像发布器
    rclcpp::TimerBase::SharedPtr timer_;                               // 帧捕获定时器
    
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

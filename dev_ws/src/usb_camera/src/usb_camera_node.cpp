#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
}

/**
 * @brief USB摄像头节点类
 * 
 * 使用FFmpeg库从USB摄像头捕获视频并发布为ROS2图像消息
 * 支持MJPG和YUYV两种像素格式
 */
class USBCameraNode : public rclcpp::Node
{
public:
    USBCameraNode() : Node("usb_camera_node")
    {
        // 声明参数
        this->declare_parameter("device_path", "/dev/video0");
        this->declare_parameter("width", 1920);
        this->declare_parameter("height", 1080);
        this->declare_parameter("fps", 30);
        this->declare_parameter("pixel_format", "mjpeg");
        this->declare_parameter("publish_compressed", true);
        this->declare_parameter("jpeg_quality", 60);
        
        // 获取参数
        device_path_ = this->get_parameter("device_path").as_string();
        width_ = this->get_parameter("width").as_int();
        height_ = this->get_parameter("height").as_int();
        fps_ = this->get_parameter("fps").as_int();
        pixel_format_ = this->get_parameter("pixel_format").as_string();
        publish_compressed_ = this->get_parameter("publish_compressed").as_bool();
        jpeg_quality_ = this->get_parameter("jpeg_quality").as_int();
        
        // 参数验证
        if (jpeg_quality_ < 1 || jpeg_quality_ > 100) {
            RCLCPP_WARN(this->get_logger(), "Invalid JPEG quality: %d, setting to 60", jpeg_quality_);
            jpeg_quality_ = 60;
        }
        
        RCLCPP_INFO(this->get_logger(), "USB Camera Node Starting");
        RCLCPP_INFO(this->get_logger(), "Device Path: %s", device_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Resolution: %dx%d", width_, height_);
        RCLCPP_INFO(this->get_logger(), "Frame Rate: %d fps", fps_);
        RCLCPP_INFO(this->get_logger(), "Pixel Format: %s", pixel_format_.c_str());
        RCLCPP_INFO(this->get_logger(), "Publish Compressed: %s", publish_compressed_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "JPEG Quality: %d", jpeg_quality_);
        
        // 创建图像发布器
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", 10);
        
        // 根据配置决定是否创建压缩图像发布器
        if (publish_compressed_) {
            compressed_image_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("camera/image_compressed", 10);
            RCLCPP_INFO(this->get_logger(), "Compressed image publisher created for topic: camera/image_compressed");
        } else {
            RCLCPP_INFO(this->get_logger(), "Compressed image publishing disabled");
        }
        
        // 初始化FFmpeg
        if (!initialize_camera()) {
            RCLCPP_ERROR(this->get_logger(), "Camera initialization failed");
            return;
        }
        
        // 创建定时器，控制发布频率
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / fps_),
            std::bind(&USBCameraNode::capture_and_publish, this)
        );
        
        RCLCPP_INFO(this->get_logger(), "Camera initialized successfully");
        
        // 初始化帧计数器和时间
        frame_count_ = 0;
        start_time_ = this->now();
    }
    
    ~USBCameraNode()
    {
        cleanup_camera();
    }

private:
    /**
     * @brief 初始化FFmpeg摄像头和编解码器
     * @return 初始化成功返回true，失败返回false
     */
    bool initialize_camera()
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
        
        // 根据配置设置像素格式
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
        
        // 打印编解码器信息
        RCLCPP_INFO(this->get_logger(), "Codec ID: %d", codec_params_->codec_id);
        RCLCPP_INFO(this->get_logger(), "Pixel Format: %d", codec_params_->format);
        RCLCPP_INFO(this->get_logger(), "Width: %d, Height: %d", codec_params_->width, codec_params_->height);
        
        // 查找解码器
        codec_ = avcodec_find_decoder(codec_params_->codec_id);
        if (!codec_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to find decoder");
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Found decoder: %s", codec_->name);
        
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
                    // 转换为OpenCV格式并发布
                    publish_frame();
                }
            }
            av_packet_unref(packet_);
        }
    }
    
    /**
     * @brief 将FFmpeg帧转换为YUV420SP格式并发布为ROS2消息
     */
    void publish_frame()
    {
        if (!frame_) return;
        
        // 创建OpenCV Mat
        cv::Mat cv_frame;
        
        // 检查帧数据
        if (frame_->linesize[0] > 0 && frame_->data[0]) {
            // 根据像素格式处理图像
            if (pixel_format_ == "mjpeg") {
                // MJPG格式处理
                RCLCPP_DEBUG(this->get_logger(), "Processing MJPG format image");
                
                // 获取帧数据大小
                int frame_size = frame_->linesize[0];
                RCLCPP_DEBUG(this->get_logger(), "MJPG frame size: %d bytes", frame_size);
                
                // 检查数据是否为空
                if (frame_size <= 0) {
                    RCLCPP_WARN(this->get_logger(), "Frame data is empty");
                    return;
                }
                
                // 检查前几个字节，判断是否为JPEG数据
                bool is_jpeg = false;
                if (frame_size >= 2) {
                    if (frame_->data[0][0] == 0xFF && frame_->data[0][1] == 0xD8) {
                        is_jpeg = true;
                        RCLCPP_DEBUG(this->get_logger(), "Detected JPEG header marker");
                    } else {
                        RCLCPP_DEBUG(this->get_logger(), "First two bytes: 0x%02X 0x%02X", 
                                   frame_->data[0][0], frame_->data[0][1]);
                    }
                }
                
                if (is_jpeg) {
                    // 查找JPEG结束标记
                    int jpeg_size = frame_size;
                    for (int i = 0; i < frame_size - 1; i++) {
                        if (frame_->data[0][i] == 0xFF && frame_->data[0][i+1] == 0xD9) {
                            jpeg_size = i + 2; // 包含JPEG结束标记
                            break;
                        }
                    }
                    
                    RCLCPP_DEBUG(this->get_logger(), "Detected JPEG size: %d bytes", jpeg_size);
                    
                    // 使用imdecode解码JPEG数据
                    std::vector<uint8_t> jpeg_data(frame_->data[0], frame_->data[0] + jpeg_size);
                    cv_frame = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);
                    
                    if (cv_frame.empty()) {
                        RCLCPP_WARN(this->get_logger(), "JPEG decoding failed");
                        return;
                    }
                } else {
                    // 如果不是JPEG格式，可能是FFmpeg已经解码为原始格式
                    RCLCPP_DEBUG(this->get_logger(), "Attempting to process raw format data");
                    
                    // 打印详细的帧信息
                    RCLCPP_DEBUG(this->get_logger(), "Frame info: linesize[0]=%d, linesize[1]=%d, linesize[2]=%d", 
                                frame_->linesize[0], frame_->linesize[1], frame_->linesize[2]);
                    RCLCPP_DEBUG(this->get_logger(), "Frame format: %d", frame_->format);
                    RCLCPP_DEBUG(this->get_logger(), "Frame width: %d, height: %d", frame_->width, frame_->height);
                    
                    // 检查FFmpeg解码器的输出格式
                    if (frame_->format == AV_PIX_FMT_YUV420P || frame_->format == 13) {
                        RCLCPP_DEBUG(this->get_logger(), "Detected YUV420P format");
                        // YUV420P格式，需要从三个平面重建图像
                        cv::Mat yuv420p(height_ * 3 / 2, width_, CV_8UC1);
                        
                        // 复制Y平面
                        memcpy(yuv420p.data, frame_->data[0], width_ * height_);
                        // 复制U平面
                        memcpy(yuv420p.data + width_ * height_, frame_->data[1], width_ * height_ / 4);
                        // 复制V平面
                        memcpy(yuv420p.data + width_ * height_ * 5 / 4, frame_->data[2], width_ * height_ / 4);
                        
                        // 转换为BGR
                        cv::cvtColor(yuv420p, cv_frame, cv::COLOR_YUV2BGR_I420);
                        
                    } else if (frame_->format == AV_PIX_FMT_YUYV422) {
                        RCLCPP_DEBUG(this->get_logger(), "Detected YUYV422 format");
                        cv::Mat temp_frame(height_, width_, CV_8UC2, frame_->data[0], frame_->linesize[0]);
                        cv::cvtColor(temp_frame, cv_frame, cv::COLOR_YUV2BGR_YUYV);
                        
                    } else if (frame_->format == AV_PIX_FMT_RGB24) {
                        RCLCPP_DEBUG(this->get_logger(), "Detected RGB24 format");
                        cv::Mat temp_frame(height_, width_, CV_8UC3, frame_->data[0], frame_->linesize[0]);
                        cv::cvtColor(temp_frame, cv_frame, cv::COLOR_RGB2BGR);
                        
                    } else if (frame_->format == AV_PIX_FMT_BGR24) {
                        RCLCPP_DEBUG(this->get_logger(), "Detected BGR24 format");
                        cv_frame = cv::Mat(height_, width_, CV_8UC3, frame_->data[0], frame_->linesize[0]);
                        
                    } else {
                        // 尝试根据linesize推断格式
                        // 对于YUV420P格式，linesize[0]可能不等于width，需要特殊处理
                        int bytes_per_pixel = 0;
                        if (frame_->linesize[0] > 0 && width_ > 0 && height_ > 0) {
                            bytes_per_pixel = frame_->linesize[0] / width_;
                        }
                        
                        RCLCPP_DEBUG(this->get_logger(), "Inferred bytes per pixel: %d (linesize[0]=%d, width=%d, height=%d)", 
                                   bytes_per_pixel, frame_->linesize[0], width_, height_);
                        
                        if (bytes_per_pixel == 1) {
                            // 单字节格式，可能是灰度图
                            cv_frame = cv::Mat(height_, width_, CV_8UC1, frame_->data[0], frame_->linesize[0]);
                            cv::cvtColor(cv_frame, cv_frame, cv::COLOR_GRAY2BGR);
                        } else if (bytes_per_pixel == 2) {
                            // 双字节格式，可能是YUYV
                            cv::Mat temp_frame(height_, width_, CV_8UC2, frame_->data[0], frame_->linesize[0]);
                            cv::cvtColor(temp_frame, cv_frame, cv::COLOR_YUV2BGR_YUYV);
                        } else if (bytes_per_pixel == 3) {
                            // 三字节格式，可能是RGB
                            cv::Mat temp_frame(height_, width_, CV_8UC3, frame_->data[0], frame_->linesize[0]);
                            cv::cvtColor(temp_frame, cv_frame, cv::COLOR_RGB2BGR);
                        } else {
                            RCLCPP_WARN(this->get_logger(), "Unrecognized format, bytes_per_pixel=%d, frame_format=%d", 
                                       bytes_per_pixel, frame_->format);
                            return;
                        }
                    }
                }
                
            } else if (pixel_format_ == "yuyv") {
                // YUYV格式处理
                RCLCPP_DEBUG(this->get_logger(), "Processing YUYV format image");
                
                int stride = frame_->linesize[0];
                int expected_yuyv_stride = width_ * 2; // YUYV格式每个像素2字节
                
                // 创建临时缓冲区，确保数据对齐
                cv::Mat temp_frame(height_, width_, CV_8UC2);
                
                // 逐行复制数据，处理可能的stride对齐问题
                for (int y = 0; y < height_; y++) {
                    const uint8_t* src_row = frame_->data[0] + y * stride;
                    uint8_t* dst_row = temp_frame.ptr<uint8_t>(y);
                    memcpy(dst_row, src_row, expected_yuyv_stride);
                }
                
                // 转换为BGR格式
                cv::cvtColor(temp_frame, cv_frame, cv::COLOR_YUV2BGR_YUYV);
                
            } else {
                // 其他格式，尝试自动检测
                RCLCPP_DEBUG(this->get_logger(), "Auto-detecting format");
                
                int stride = frame_->linesize[0];
                int bytes_per_pixel = stride / width_;
                
                if (bytes_per_pixel == 1) {
                    // 单字节格式，可能是灰度图
                    cv_frame = cv::Mat(height_, width_, CV_8UC1, frame_->data[0], stride);
                    cv::cvtColor(cv_frame, cv_frame, cv::COLOR_GRAY2BGR);
                } else if (bytes_per_pixel == 2) {
                    // 双字节格式，可能是YUYV
                    cv::Mat temp_frame(height_, width_, CV_8UC2, frame_->data[0], stride);
                    cv::cvtColor(temp_frame, cv_frame, cv::COLOR_YUV2BGR_YUYV);
                } else if (bytes_per_pixel == 3) {
                    // 三字节格式，可能是RGB
                    cv::Mat temp_frame(height_, width_, CV_8UC3, frame_->data[0], stride);
                    cv::cvtColor(temp_frame, cv_frame, cv::COLOR_RGB2BGR);
                } else {
                    RCLCPP_WARN(this->get_logger(), "Unrecognized format, bytes_per_pixel=%d", bytes_per_pixel);
                    return;
                }
            }
            
            if (cv_frame.empty()) {
                RCLCPP_WARN(this->get_logger(), "Image conversion failed");
                return;
            }
            
            // 确保图像是彩色的
            if (cv_frame.channels() == 1) {
                cv::cvtColor(cv_frame, cv_frame, cv::COLOR_GRAY2BGR);
            }
            
        } else {
            RCLCPP_WARN(this->get_logger(), "Invalid frame data");
            return;
        }
        
        // 将BGR转换为YUV420SP (NV12)格式
        cv::Mat yuv420sp_frame;
        cv::cvtColor(cv_frame, yuv420sp_frame, cv::COLOR_BGR2YUV_I420);
        
        // 重新排列为YUV420SP (NV12)格式
        // YUV420SP格式: Y平面 + UV交错平面
        int y_size = width_ * height_;
        int uv_size = y_size / 2;
        
        cv::Mat nv12_frame(height_ * 3 / 2, width_, CV_8UC1);
        
        // 复制Y平面
        memcpy(nv12_frame.data, yuv420sp_frame.data, y_size);
        
        // 重新排列UV平面为交错格式
        uint8_t* uv_plane = nv12_frame.data + y_size;
        const uint8_t* u_plane = yuv420sp_frame.data + y_size;
        const uint8_t* v_plane = yuv420sp_frame.data + y_size + uv_size / 2;
        
        for (int i = 0; i < uv_size / 2; i++) {
            uv_plane[i * 2] = u_plane[i];     // U分量
            uv_plane[i * 2 + 1] = v_plane[i]; // V分量
        }
        
        // 创建ROS图像消息，使用yuv420sp编码
        auto ros_image = cv_bridge::CvImage(std_msgs::msg::Header(), "yuv420sp", nv12_frame);
        ros_image.header.stamp = this->now();
        ros_image.header.frame_id = "camera_link";
        
        // 发布图像
        image_pub_->publish(*ros_image.toImageMsg());
        
        // 根据配置决定是否发布压缩图像
        if (publish_compressed_) {
            // 创建并发布压缩图像 (JPEG格式)
            auto compressed_msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
            compressed_msg->header.stamp = this->now();
            compressed_msg->header.frame_id = "camera_link";
            compressed_msg->format = "jpeg";
            
            // 将BGR图像编码为JPEG格式
            std::vector<uchar> jpeg_buffer;
            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            compression_params.push_back(jpeg_quality_);
            
            bool success = cv::imencode(".jpg", cv_frame, jpeg_buffer, compression_params);
            if (success) {
                compressed_msg->data = jpeg_buffer;
                compressed_image_pub_->publish(*compressed_msg);
                RCLCPP_DEBUG(this->get_logger(), "Published compressed image (JPEG quality: %d, size: %zu bytes)", 
                            jpeg_quality_, jpeg_buffer.size());
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to encode image to JPEG format");
            }
        }
        
        // 更新帧计数器和显示信息
        frame_count_++;
        auto current_time = this->now();
        auto elapsed = (current_time - start_time_).seconds();
        
        if (frame_count_ % 30 == 0) {  // 每30帧显示一次统计信息
            double actual_fps = frame_count_ / elapsed;
            RCLCPP_INFO(this->get_logger(), "Published %d frames, Runtime: %.1f seconds, Actual FPS: %.1f fps", 
                       frame_count_, elapsed, actual_fps);
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Published YUV420SP image: %dx%d (frame #%d)", 
                     nv12_frame.cols, nv12_frame.rows, frame_count_);
    }
    
    /**
     * @brief 清理FFmpeg资源
     */
    void cleanup_camera()
    {
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
    }
    
    // 成员变量
    std::string device_path_;           // 摄像头设备路径
    int width_;                         // 图像宽度
    int height_;                        // 图像高度
    int fps_;                           // 帧率
    std::string pixel_format_;          // 像素格式 (mjpeg 或 yuyv)
    bool publish_compressed_;           // 是否发布压缩图像
    int jpeg_quality_;                  // JPEG压缩质量 (1-100)
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;                       // 图像发布器
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_image_pub_;  // 压缩图像发布器
    rclcpp::TimerBase::SharedPtr timer_;                                                    // 帧捕获定时器
    
    // 帧计数器和时间统计
    int frame_count_;                   // 已发布的总帧数
    rclcpp::Time start_time_;           // FPS计算的开始时间
    
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
    
    auto node = std::make_shared<USBCameraNode>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
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
        
        // 获取参数
        device_path_ = this->get_parameter("device_path").as_string();
        width_ = this->get_parameter("width").as_int();
        height_ = this->get_parameter("height").as_int();
        fps_ = this->get_parameter("fps").as_int();
        pixel_format_ = this->get_parameter("pixel_format").as_string();
        
        RCLCPP_INFO(this->get_logger(), "USB摄像头节点启动");
        RCLCPP_INFO(this->get_logger(), "设备路径: %s", device_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "分辨率: %dx%d", width_, height_);
        RCLCPP_INFO(this->get_logger(), "帧率: %d fps", fps_);
        RCLCPP_INFO(this->get_logger(), "像素格式: %s", pixel_format_.c_str());
        
        // 创建图像发布器
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", 10);
        
        // 初始化FFmpeg
        if (!initialize_camera()) {
            RCLCPP_ERROR(this->get_logger(), "摄像头初始化失败");
            return;
        }
        
        // 创建定时器，控制发布频率
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000 / fps_),
            std::bind(&USBCameraNode::capture_and_publish, this)
        );
        
        RCLCPP_INFO(this->get_logger(), "摄像头初始化成功");
        
        // 初始化帧计数器和时间
        frame_count_ = 0;
        start_time_ = this->now();
    }
    
    ~USBCameraNode()
    {
        cleanup_camera();
    }

private:
    bool initialize_camera()
    {
        // 注册FFmpeg设备
        avdevice_register_all();
        
        // 创建格式上下文
        format_context_ = avformat_alloc_context();
        if (!format_context_) {
            RCLCPP_ERROR(this->get_logger(), "无法分配格式上下文");
            return false;
        }
        
        // 设置输入格式
        input_format_ = av_find_input_format("video4linux2");
        if (!input_format_) {
            RCLCPP_ERROR(this->get_logger(), "无法找到video4linux2输入格式");
            return false;
        }
        
        // 设置选项
        AVDictionary* options = nullptr;
        std::string video_size = std::to_string(width_) + "x" + std::to_string(height_);
        std::string framerate = std::to_string(fps_);
        
        av_dict_set(&options, "video_size", video_size.c_str(), 0);
        av_dict_set(&options, "framerate", framerate.c_str(), 0);
        
        // 使用正确的选项名称
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
            RCLCPP_ERROR(this->get_logger(), "无法打开摄像头: %s", errbuf);
            av_dict_free(&options);
            return false;
        }
        
        av_dict_free(&options);
        
        // 查找流信息
        if (avformat_find_stream_info(format_context_, nullptr) < 0) {
            RCLCPP_ERROR(this->get_logger(), "无法找到流信息");
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
            RCLCPP_ERROR(this->get_logger(), "无法找到视频流");
            return false;
        }
        
        // 获取编解码器参数
        codec_params_ = format_context_->streams[video_stream_index_]->codecpar;
        
        // 查找解码器
        codec_ = avcodec_find_decoder(codec_params_->codec_id);
        if (!codec_) {
            RCLCPP_ERROR(this->get_logger(), "无法找到解码器");
            return false;
        }
        
        // 创建解码器上下文
        codec_context_ = avcodec_alloc_context3(codec_);
        if (!codec_context_) {
            RCLCPP_ERROR(this->get_logger(), "无法分配解码器上下文");
            return false;
        }
        
        // 将参数复制到解码器上下文
        if (avcodec_parameters_to_context(codec_context_, codec_params_) < 0) {
            RCLCPP_ERROR(this->get_logger(), "无法复制编解码器参数");
            return false;
        }
        
        // 打开解码器
        if (avcodec_open2(codec_context_, codec_, nullptr) < 0) {
            RCLCPP_ERROR(this->get_logger(), "无法打开解码器");
            return false;
        }
        
        // 分配帧和数据包
        frame_ = av_frame_alloc();
        packet_ = av_packet_alloc();
        
        if (!frame_ || !packet_) {
            RCLCPP_ERROR(this->get_logger(), "无法分配帧或数据包");
            return false;
        }
        
        return true;
    }
    
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
                    RCLCPP_WARN(this->get_logger(), "发送数据包失败");
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
    
    void publish_frame()
    {
        if (!frame_) return;
        
        // 创建OpenCV Mat
        cv::Mat cv_frame;
        
        // 检查帧数据
        if (frame_->linesize[0] > 0 && frame_->data[0]) {
            int stride = frame_->linesize[0];
            int expected_yuyv_stride = width_ * 2; // YUYV格式每个像素2字节
            
            RCLCPP_DEBUG(this->get_logger(), "帧信息: linesize=%d, expected_yuyv=%d", stride, expected_yuyv_stride);
            
            if (stride >= expected_yuyv_stride) {
                // 可能是YUYV格式
                RCLCPP_DEBUG(this->get_logger(), "检测到YUYV格式");
                
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
            } else if (stride == width_) {
                // 可能是灰度格式或其他单字节格式
                RCLCPP_DEBUG(this->get_logger(), "检测到单字节格式，尝试MJPG解码");
                
                // 尝试作为MJPG解码
                cv_frame = cv::imdecode(cv::Mat(1, stride * height_, CV_8UC1, frame_->data[0]), cv::IMREAD_COLOR);
                
                if (cv_frame.empty()) {
                    RCLCPP_DEBUG(this->get_logger(), "MJPG解码失败，尝试其他格式");
                    // 如果MJPG失败，尝试其他可能的格式
                    cv_frame = cv::Mat(height_, width_, CV_8UC1, frame_->data[0], stride);
                    cv::cvtColor(cv_frame, cv_frame, cv::COLOR_GRAY2BGR);
                }
            } else {
                // 其他格式，尝试直接创建Mat
                RCLCPP_DEBUG(this->get_logger(), "未知格式，尝试直接处理");
                
                // 计算每像素字节数
                int bytes_per_pixel = stride / width_;
                if (bytes_per_pixel > 0 && bytes_per_pixel <= 4) {
                    int cv_type = CV_8UC(bytes_per_pixel);
                    cv_frame = cv::Mat(height_, width_, cv_type, frame_->data[0], stride);
                    
                    // 根据类型转换颜色空间
                    if (bytes_per_pixel == 1) {
                        cv::cvtColor(cv_frame, cv_frame, cv::COLOR_GRAY2BGR);
                    } else if (bytes_per_pixel == 2) {
                        cv::cvtColor(cv_frame, cv_frame, cv::COLOR_YUV2BGR_YUYV);
                    } else if (bytes_per_pixel == 3) {
                        cv::cvtColor(cv_frame, cv_frame, cv::COLOR_RGB2BGR);
                    } else if (bytes_per_pixel == 4) {
                        cv::cvtColor(cv_frame, cv_frame, cv::COLOR_RGBA2BGR);
                    }
                } else {
                    RCLCPP_WARN(this->get_logger(), "无法确定像素格式，bytes_per_pixel=%d", bytes_per_pixel);
                    return;
                }
            }
            
            if (cv_frame.empty()) {
                RCLCPP_WARN(this->get_logger(), "图像转换失败");
                return;
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "帧数据无效");
            return;
        }
        
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
            RCLCPP_INFO(this->get_logger(), "已发布 %d 帧, 运行时间: %.1f秒, 实际帧率: %.1f fps", 
                       frame_count_, elapsed, actual_fps);
        }
        
        RCLCPP_DEBUG(this->get_logger(), "发布图像: %dx%d (帧 #%d)", 
                     cv_frame.cols, cv_frame.rows, frame_count_);
    }
    
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
    std::string device_path_;
    int width_;
    int height_;
    int fps_;
    std::string pixel_format_;
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // 帧计数器和时间统计
    int frame_count_;
    rclcpp::Time start_time_;
    
    // FFmpeg相关
    AVFormatContext* format_context_;
    AVInputFormat* input_format_;
    AVCodecParameters* codec_params_;
    AVCodec* codec_;
    AVCodecContext* codec_context_;
    AVFrame* frame_;
    AVPacket* packet_;
    int video_stream_index_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<USBCameraNode>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}

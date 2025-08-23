cmake_minimum_required(VERSION 3.10)

# 项目名称
project(usb_camera_demo)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找FFmpeg相关库
find_package(PkgConfig REQUIRED)

# 查找FFmpeg库
pkg_check_modules(FFMPEG REQUIRED
    libavcodec
    libavformat
    libavutil
    libavdevice
    libswscale
)

# 创建可执行文件
add_executable(main main.cpp)

# 链接FFmpeg库
target_link_libraries(main ${FFMPEG_LIBRARIES})

# 设置包含目录
target_include_directories(main PRIVATE ${FFMPEG_INCLUDE_DIRS})

# 设置编译选项
target_compile_options(main PRIVATE ${FFMPEG_CFLAGS_OTHER})

# 设置链接选项
target_link_options(main PRIVATE ${FFMPEG_LDFLAGS_OTHER})

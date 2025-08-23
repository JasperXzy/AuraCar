cmake_minimum_required(VERSION 3.8)
project(usb_camera VERSION 0.0.1 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译器特定选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
  add_compile_options(-fPIC)
elseif(MSVC)
  add_compile_options(/W4 /O2)
endif()

# 设置编译类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 查找ROS2依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# 查找OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV not found. Please install OpenCV development packages.")
endif()

# 查找FFmpeg库
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  # 尝试使用pkg-config查找FFmpeg
  pkg_check_modules(FFMPEG QUIET libavcodec libavformat libavutil libavdevice)
endif()

# 如果pkg-config失败，手动查找FFmpeg库
if(NOT FFMPEG_FOUND)
  find_library(AVCODEC_LIBRARY NAMES avcodec)
  find_library(AVFORMAT_LIBRARY NAMES avformat)
  find_library(AVUTIL_LIBRARY NAMES avutil)
  find_library(AVDEVICE_LIBRARY NAMES avdevice)
  
  if(AVCODEC_LIBRARY AND AVFORMAT_LIBRARY AND AVUTIL_LIBRARY AND AVDEVICE_LIBRARY)
    set(FFMPEG_LIBRARIES ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY})
    set(FFMPEG_FOUND TRUE)
  else()
    message(FATAL_ERROR "FFmpeg libraries not found. Please install FFmpeg development packages.")
  endif()
endif()

# 创建可执行文件
add_executable(usb_camera_node src/usb_camera_node.cpp)

# 设置目标属性
set_target_properties(usb_camera_node PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# ROS2依赖
ament_target_dependencies(usb_camera_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
)

# 链接库
target_link_libraries(usb_camera_node
  ${OpenCV_LIBS}
  ${FFMPEG_LIBRARIES}
)

# 包含目录
target_include_directories(usb_camera_node PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)

# 如果FFmpeg有包含目录，也添加
if(FFMPEG_INCLUDE_DIRS)
  target_include_directories(usb_camera_node PUBLIC ${FFMPEG_INCLUDE_DIRS})
endif()

# 编译定义
target_compile_definitions(usb_camera_node PRIVATE
  $<$<CONFIG:Debug>:_DEBUG>
  $<$<CONFIG:Release>:NDEBUG>
)

# 安装目标
install(TARGETS
  usb_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装启动文件
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# 安装配置文件
install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

# 打印配置信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "FFmpeg libraries: ${FFMPEG_LIBRARIES}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

ament_package()

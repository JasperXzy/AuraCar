cmake_minimum_required(VERSION 3.8)
project(lane_detection LANGUAGES CXX)

# 使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 依赖的 ROS 2 与第三方库
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
# foxglove_msgs 提供 ImageAnnotations/PointsAnnotation
find_package(foxglove_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Ascend CANN 相关路径
set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    # 默认安装路径
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub")
endif ()

set(THIRDPART $ENV{THIRDPART_PATH})
if (NOT DEFINED ENV{THIRDPART_PATH})
    set(THIRDPART "/usr/local/Ascend/thirdpart/aarch64")
endif ()

# 查找 Ascend 运行时库/第三方 acllite 封装库
find_library(ACL_LIBRARY NAMES ascendcl PATHS ${LIB_PATH} NO_DEFAULT_PATH)
find_library(ACL_DVPP_LIBRARY NAMES acl_dvpp PATHS ${LIB_PATH} NO_DEFAULT_PATH)
find_library(ACLLITE_LIBRARY NAMES acllite PATHS ${THIRDPART}/lib NO_DEFAULT_PATH)

if(NOT ACL_LIBRARY OR NOT ACL_DVPP_LIBRARY OR NOT ACLLITE_LIBRARY)
  message(WARNING "Ascend libraries not found. Ensure DDK_PATH, NPU_HOST_LIB, THIRDPART_PATH are set.")
endif()

# 头文件搜索路径
include_directories(
  include
  ${INC_PATH}/runtime/include/
  ${THIRDPART}/include/acllite/
)

# 推理与后处理静态库
add_library(clrnet_infer
  src/clrnet_infer.cpp
  src/postprocess.cpp
)

# 启用 DVPP 接口定义
target_compile_definitions(clrnet_infer PRIVATE ENABLE_DVPP_INTERFACE=1)

# 链接 Ascend/CANN 运行时、acllite、OpenCV 以及基础系统库
target_link_libraries(clrnet_infer
  ${ACL_LIBRARY}
  ${ACL_DVPP_LIBRARY}
  ${ACLLITE_LIBRARY}
  ${OpenCV_LIBS}
  stdc++
  dl
  rt
)

ament_target_dependencies(clrnet_infer rclcpp sensor_msgs)

# 节点可执行程序
add_executable(lane_detection_node src/lane_detection_node.cpp)
target_include_directories(lane_detection_node PUBLIC include)
target_compile_definitions(lane_detection_node PRIVATE ENABLE_DVPP_INTERFACE=1)
target_link_libraries(lane_detection_node clrnet_infer ${OpenCV_LIBS})
ament_target_dependencies(lane_detection_node rclcpp sensor_msgs foxglove_msgs std_msgs)

# 安装可执行文件与启动文件
install(TARGETS lane_detection_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME})

ament_package()

## 第三方依赖安装

设置环境变量，配置程序编译依赖的头文件、库文件路径，"$HOME/Ascend" 替换为实际的 Ascend CANN Toolkit 安装路径

```
export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
export THIRDPART_PATH=${DDK_PATH}/thirdpart
export LD_LIBRARY_PATH=${THIRDPART_PATH}/lib:$LD_LIBRARY_PATH
```

创建 THIRDPART_PATH 路径

```
mkdir -p ${THIRDPART_PATH}
```

- acllite

    注：源码安装 ffmpeg 主要是为了 acllite 库的安装
    执行以下命令安装 x264

    ```
    # 下载x264
    cd ${HOME}
    git clone https://code.videolan.org/videolan/x264.git
    cd x264
    # 安装x264
    ./configure --enable-shared --disable-asm
    make
    sudo make install
    sudo cp /usr/local/lib/libx264.so.164 /lib
    ```   
    执行以下命令安装 ffmpeg

    ```
    # 下载 ffmpeg
    cd ${HOME}
    wget http://www.ffmpeg.org/releases/ffmpeg-4.1.3.tar.gz --no-check-certificate
    tar -zxvf ffmpeg-4.1.3.tar.gz
    cd ffmpeg-4.1.3
    # 安装 ffmpeg
    ./configure --enable-shared --enable-pic --enable-static --disable-x86asm --enable-libx264 --enable-gpl --prefix=${THIRDPART_PATH}
    make -j8
    make install
    ```   
   执行以下命令安装 acllite

    ```
    cd ${HOME}/samples/inference/acllite/cplusplus
    make
    make install
    ```   
    </details> 

- opencv

  执行以下命令安装 opencv (注:确保是3.x版本)
  ```
  sudo apt-get install libopencv-dev
  ```   

## 样例运行

- 数据准备

  已在 `demo/yolo11_seg/data` 目录提供示例图片，可自行替换

- ATC 模型转换

  将 YOLO11 分割原始模型转换为昇腾离线模型（.om），并放在 `demo/yolo11_seg/model` 目录

  示例（以 yolo11n-seg 为例，输入大小 640x640，请按实际模型/芯片版本调整）：
  ```
  cd ./demo/yolo11_seg/model
  # 准备文件
  # yolo11n-seg.onnx 与 aipp.cfg 请根据实际情况准备

  atc --model=yolo11n-seg.onnx --framework=5 --output=yolo11n-seg --input_shape="images:1,3,640,640"  --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg
  ```

- 样例编译

  执行以下脚本进行编译：
  ```
  cd ./demo/yolo11_seg/scripts
  bash sample_build.sh
  ```
  编译目录：`demo/yolo11_seg/build`；可执行文件输出：`demo/yolo11_seg/out/main`

- 样例运行

  执行运行脚本：
  ```
  bash sample_run.sh
  ```

- 样例结果

   运行完成后，会在样例工程的out目录下生成推理后的图片：
   - `out_X.jpg`: 显示检测框和标签的结果图
   - `mask_X.jpg`: 显示分割mask叠加的结果图

## YOLO11分割模型说明

YOLO11分割模型相比检测模型的主要区别：

1. **输出结构**：
   - 输出0: 检测结果 [1, 84, 8400] (4个坐标 + 80个类别置信度)
   - 输出1: mask原型 [1, 32, 8400] (32个mask原型系数)

2. **分割处理**：
   - 使用mask原型系数与检测框结合生成最终的分割mask
   - 支持实例分割，每个检测到的对象都有对应的分割mask

3. **可视化**：
   - 同时输出检测框和分割mask的可视化结果
   - 使用不同颜色区分不同类别的分割区域

## 常见问题排查

- 编译时报错找不到 acllite 头文件或库

  - 现象：执行 `bash scripts/sample_build.sh` 时报如下错误之一：
    ```
    fatal error: AclLite*.h: No such file or directory
    cannot find -lacllite 或找不到 libacllite.so
    ```

  - 原因：未执行 `cd ${HOME}/samples/inference/acllite/cplusplus && make && make install`，或 `THIRDPART_PATH` 未指向安装产物位置

  - 解决：
   1) 设置环境变量并创建目录：
        ```
        export DDK_PATH="$HOME/Ascend/ascend-toolkit/latest"
        export THIRDPART_PATH="${THIRDPART_PATH_DEFAULT}"
        mkdir -p "$THIRDPART_PATH"
        ```
   2) 编译安装 acllite：
        ```
        cd "$HOME/samples/inference/acllite/cplusplus"
        make
        make install
        ```
   3) 确认产物存在：`${THIRDPART_PATH}/include/acllite` 与 `${THIRDPART_PATH}/lib/libacllite.so`
   4) 重新执行：
        ```
        cd ./demo/yolo11_seg/scripts
        bash sample_build.sh
        ```

- 模型转换问题

  - 确保使用YOLO11分割模型（yolo11n-seg.onnx）而不是检测模型
  - 检查模型输入输出维度是否正确
  - 确认aipp.cfg配置文件适用于分割模型

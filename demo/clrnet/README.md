# CLRNet C++ Demo

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

  已在 `demo/clrnet/data` 目录提供示例图片，可自行替换

- ATC 模型转换

  将 CLRNet 原始模型转换为昇腾离线模型（.om），并放在 `demo/clrnet/model` 目录

  示例（以 clrnet 为例，输入大小 320x800，请按实际模型/芯片版本调整）：
  ```
  cd ./demo/clrnet/model
  # 准备文件
  # clrnet.onnx 与 aipp.cfg 请根据实际情况准备

  atc --model=clrnet.onnx --framework=5 --output=clrnet --input_shape="images:1,3,320,800"  --soc_version=Ascend310B1
  ```

- 样例编译

  执行以下脚本进行编译：
  ```
  cd ./demo/clrnet/scripts
  bash sample_build.sh
  ```
  编译目录：`demo/clrnet/build`；可执行文件输出：`demo/clrnet/out/main`

- 样例运行

  执行运行脚本：
  ```
  bash sample_run.sh
  ```

- 样例结果

   运行完成后，会在样例工程的out目录下生成推理后的图片：
   - `clrnet_result_X.jpg`: 显示车道线检测结果图，绿色圆点标记检测到的车道线点

## CLRNet车道线检测模型说明

CLRNet车道线检测模型的主要特点：

1. **输入结构**：
   - 输入: [1, 3, 320, 800] - 经过预处理的图像
   - 图像预处理：裁剪底部区域、缩放到指定尺寸、归一化到[0,1]

2. **输出结构**：
   - 输出: [1, 192, 78] - 车道线检测结果
   - 包含置信度、起始位置、角度、长度和偏移量信息

3. **后处理流程**：
   - 置信度计算：使用softmax计算车道线置信度
   - 阈值过滤：过滤低置信度的车道线
   - NMS处理：去除重复的车道线
   - 坐标解码：将模型输出转换为图像坐标

4. **可视化**：
   - 在原始图像上用绿色圆点标记检测到的车道线点
   - 支持批量图像处理

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
        cd ./demo/clrnet/scripts
        bash sample_build.sh
        ```

- 模型转换问题

  - 确保使用CLRNet车道线检测模型（clrnet.onnx）
  - 检查模型输入输出维度是否正确
  - 确认aipp.cfg配置文件适用于车道线检测模型
  - 注意输入尺寸为320x800，与其他模型不同

- 运行时问题

  - 确保模型文件`clrnet.om`已正确放置在`model/`目录
  - 检查测试图像是否已放置在`data/`目录
  - 确认环境变量设置正确

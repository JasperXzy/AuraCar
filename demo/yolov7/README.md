## 第三方依赖安装

设置环境变量，配置程序编译依赖的头文件、库文件路径，“$HOME/Ascend” 替换 “Ascend-cann-toolkit” 包的实际安装路径

   ```
    export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
    export THIRDPART_PATH=${DDK_PATH}/thirdpart
    export LD_LIBRARY_PATH=${THIRDPART_PATH}/lib:$LD_LIBRARY_PATH
   ```
   创建THIRDPART_PATH路径

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

    从以下链接获取该样例的输入图片，放在data目录下
        
    ```    
    cd ./demo/yolov7/data
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1_1024_683.jpg
    ```

  - ATC模型转换

    将 YOLOV7 原始模型转换为适配昇腾310处理器的离线模型（\*.om文件），放在model路径下

    ```
    # 原始模型下载
    cd ./demo/yolov7/model
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/yolov7/yolov7x.onnx
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/yolov7/aipp.cfg
    atc --model=yolov7x.onnx --framework=5 --output=yolov7x --input_shape="images:1,3,640,640"  --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg
    ```

  - 样例编译

    执行以下命令，执行编译脚本，开始样例编译
    ```
    cd ./demo/yolov7/scripts
    bash sample_build.sh
    ```
  - 样例运行

    执行运行脚本，开始样例运行
    ```
    bash sample_run.sh
    ```
  - 样例结果展示
    
   运行完成后，会在样例工程的out目录下生成推理后的图片，显示对比结果如下所示
   ![输入图片说明](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/yolov7/out_dog.jpg "image-20211028101534905.png")

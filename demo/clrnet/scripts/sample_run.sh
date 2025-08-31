#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

# 设置环境变量
export DDK_PATH=${DDK_PATH:-$HOME/Ascend/ascend-toolkit/latest}
export NPU_HOST_LIB=${NPU_HOST_LIB:-$DDK_PATH/runtime/lib64/stub}
export THIRDPART_PATH=${THIRDPART_PATH:-${DDK_PATH}/thirdpart}
export LD_LIBRARY_PATH=${THIRDPART_PATH}/lib:$LD_LIBRARY_PATH

# 确保THIRDPART_PATH目录存在
if [ -n "$THIRDPART_PATH" ]; then
  mkdir -p "${THIRDPART_PATH}"
  echo "[INFO] Created THIRDPART_PATH directory: ${THIRDPART_PATH}"
fi

# 如果THIRDPART_PATH未设置，使用默认路径
if [ -z "$THIRDPART_PATH" ]; then
  THIRDPART_PATH_DEFAULT=${THIRDPART_PATH:-/usr/local/Ascend/thirdpart/aarch64}
  mkdir -p "${THIRDPART_PATH_DEFAULT}"
  echo "[INFO] Created default THIRDPART_PATH directory: ${THIRDPART_PATH_DEFAULT}"
fi

# 显示环境变量信息
echo "[INFO] Environment variables:"
echo "  DDK_PATH: ${DDK_PATH}"
echo "  NPU_HOST_LIB: ${NPU_HOST_LIB}"
echo "  THIRDPART_PATH: ${THIRDPART_PATH}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

echo "[INFO] The sample starts to run"
running_command="./main"
cd ${ScriptPath}/../out || exit 1
${running_command}
if [ $? -ne 0 ];then
    echo "[INFO] The program runs failed"
else
    echo "[INFO] The program runs successfully"
fi

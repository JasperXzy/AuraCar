#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
ModelPath="${ScriptPath}/../model"
THIRDPART_PATH_DEFAULT=${THIRDPART_PATH:-/usr/local/Ascend/thirdpart/aarch64}

function build()
{
  BUILD_DIR="${ScriptPath}/../build"
  mkdir -p "${BUILD_DIR}"
  cd "${BUILD_DIR}" || exit 1

  cmake ../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
  if [ $? -ne 0 ]; then
    echo "[ERROR] cmake error, Please check your environment!"
    return 1
  fi

  make
  if [ $? -ne 0 ]; then
    echo "[ERROR] build failed, Please check your environment!"
    return 1
  fi
  cd - > /dev/null
}

function main()
{
  echo "[INFO] Sample preparation"

  ret=`find ${ModelPath} -maxdepth 1 -name yolov8n-seg.om 2> /dev/null`

  if [[ ${ret} ]];then
    echo "[INFO] The yolov8n-seg.om already exists. Start building"
  else
    echo "[ERROR] yolov8n-seg.om does not exist, please follow the README to convert the model and place it in the correct position!"
    return 1
  fi
  if [ ! -d "${THIRDPART_PATH_DEFAULT}/include/acllite" ] || [ ! -e "${THIRDPART_PATH_DEFAULT}/lib/libacllite.so" ]; then
    echo "[ERROR] acllite not found under THIRDPART_PATH=${THIRDPART_PATH_DEFAULT}"
    echo "        Please install acllite and set THIRDPART_PATH accordingly. Typical steps:"
    echo "          export DDK_PATH=\"$HOME/Ascend/ascend-toolkit/latest\""
    echo "          export THIRDPART_PATH=\"${THIRDPART_PATH_DEFAULT}\"   # or your actual install dir"
    echo "          mkdir -p \"$THIRDPART_PATH\""
    echo "          # build and install acllite"
    echo "          cd \"$HOME/samples/inference/acllite/cplusplus\" && make && make install"
    echo "        Or adjust THIRDPART_PATH to where libacllite.so and include/acllite are installed."
    return 1
  else
    echo "[INFO] Found acllite at ${THIRDPART_PATH_DEFAULT}"
  fi

  build
  if [ $? -ne 0 ];then
    return 1
  fi
    
  echo "[INFO] Sample preparation is complete"
}
main

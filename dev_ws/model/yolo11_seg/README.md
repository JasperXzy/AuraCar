```bash
atc --model=yolo11n-seg.onnx \
    --framework=5 \
    --output=yolo11n-seg \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --insert_op_conf=aipp.cfg \
    --soc_version=Ascend310B1
```

```bash
atc --model=clrnet.onnx \
    --framework=5 \
    --output=clrnet \
    --input_format=NCHW \
    --input_shape="images:1,3,320,800" \
    --insert_op_conf=aipp.cfg \
    --soc_version=Ascend310B1
```

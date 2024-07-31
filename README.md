## Change batch size for static shape/dynamic shape onnx model

### Install

```shell
pip install -r requirements.txt
```

### Change dynamic shape onnx model to fixed shape

```shell
python make_onnx_dynamic_shape_to_fixed.py --onnx_path yolov5n.onnx   --input_name 'images' --input_shape 6,3,640,640
```

### Change static shape onnx model

```shell
python change_batch_size_for_static_shape.py --onnx_path yolov5n_bs6.onnx --batch_size 2
```

### Ref:

[Changing Batch SIze](https://github.com/onnx/onnx/issues/2182)
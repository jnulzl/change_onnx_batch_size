import os
import sys
import argparse
import onnx
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed
import onnxsim

def change_input_dim():
    parser = argparse.ArgumentParser(description="Making dynamic onnx input shapes fixed")

    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="The path of onnx model",
    )
   
    parser.add_argument(
        "--input_name",
        type=str,
        required=True,
        help="Model input name to replace shape of. Provide input_shape if specified.",
    )
    parser.add_argument(
        "--input_shape",
        type=lambda x: [int(i) for i in x.split(",")],
        required=True,
        help="Shape to use for input_shape. Provide comma separated list for the shape. "
        "All values must be > 0. e.g. --input_shape 1,3,256,256",
    )
    
    args = parser.parse_args()
    model = onnx.load(args.onnx_path)
    make_input_shape_fixed(model.graph, args.input_name, args.input_shape)
    fix_output_shapes(model)
        
    print("Simplifying...")    
    model_opt, check = onnxsim.simplify(model)
    print("Finish! Here is the difference:")
    onnxsim.model_info.print_simplifying_info(model, model_opt)
    out_onnx_path = args.onnx_path.replace(".onnx", "_bs%d.onnx"%(args.input_shape[0]))    
    onnx.save(model_opt, out_onnx_path)
      

if __name__ == '__main__':
    change_input_dim()
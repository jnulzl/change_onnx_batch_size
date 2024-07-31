import argparse
import struct
import onnx
import onnxsim

def rebatch(model, batch_size):
    graph = model.graph

    # Change batch size in input, output and value_info
    all_tensors = list(graph.input) + list(graph.value_info) + list(graph.output)
    for tensor in all_tensors:
        tensor.type.tensor_type.shape.dim[0].dim_param = str(batch_size)

    # Set dynamic batch size in reshapes (-1)
    for node in  graph.node:
        if node.op_type != 'Reshape':
            continue
        for init in graph.initializer:
            # node.input[1] is expected to be a reshape
            if init.name != node.input[1]:
                continue
            # Shape is stored as a list of ints
            if len(init.int64_data) > 0:
                # This overwrites bias nodes' reshape shape but should be fine
                init.int64_data[0] = -1
            # Shape is stored as bytes
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into('q', shape, 0, -1)
                init.raw_data = bytes(shape)

    # onnx.save(model, outfile)
    return model


def main():
    parser = argparse.ArgumentParser(description="Change onnx batch size")

    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="The path of onnx model",
    )
   
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="New batch size",
    )
    
    args = parser.parse_args()
    model = onnx.load(args.onnx_path)
    rebatch(model, args.batch_size)
    
    print("Simplifying...")
    model_opt, check = onnxsim.simplify(model)
    print("Finish! Here is the difference:")
    onnxsim.model_info.print_simplifying_info(model, model_opt)
    
    outfile = args.onnx_path.replace(".onnx", "_bs%d.onnx"%(args.batch_size))
    onnx.save(model_opt, outfile)


if __name__ == '__main__':
    main()
# baseline cnn model for mnist
import sys
import numpy as np
import onnx
import onnxruntime


def onnxruntime_mod(input_path, input_file, output_prefix, input_tensor_name):
    """
    Modifies ONNX model by adding some internal parameter to the model's output vector.

    :param input_path: Path to where input .onnx file resides
    :param input_file: Input file name
    :param output_prefix: Prefix added to output .onnx file
    :param input_tensor_name: ONNX identifier of input to be added to ONNX output

    :return: None
    """
    model = onnx.load(input_path + input_file)
    intermediate_tensor_name = input_tensor_name
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = intermediate_tensor_name
    model.graph.output.extend([intermediate_layer_value_info])
    onnx.save(model, input_path + output_prefix + input_file)


def onnxruntime_test(model_path, input_name):
    """
    Runs inference on MNIST model provided for one test input.

    :param model_path: Path to model .onnx file, including file name
    :param input_name: ONNX identifier of entry point

    :return: Inference result
    """
    session = onnxruntime.InferenceSession(model_path)

    image_input = np.zeros([28, 28], dtype=float)
    image_input[:, 13:16] = 1.0
    # image_input[14:15, :] = 1.0
    print(image_input)
    return session.run(None, {input_name: [[image_input]]})


def print_pretty(arr, nrows, ncols):
    """
    Formatting helper function for 2 or 4 dimensional Numpy arrays.
    Only prints two dimensions.

    :param arr: Numpy array to be printed.
    :param nrows: Number of rows of output
    :param ncols: Number of cols of output

    :return: None
    """
    if arr.ndim == 4:
        for row in range(nrows):
            buf = ""
            for col in range(ncols):
                buf += str(arr[0, 0, row, col]) + " "
            print(buf)
    elif arr.ndim == 2:
        for row in range(nrows):
            buf = ""
            for col in range(ncols):
                buf += str(arr[row, col]) + " "
            print(buf)


# entry point, run the test harness
if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    # only run the following line once
    # onnxruntime_mod("/home/fdm/Documents/BA/git/garbled-inference/models/", "mnist-8.onnx", "reshape2_", "Parameter193_reshape1")

    ans = onnxruntime_test("/home/fdm/Documents/BA/git/garbled-inference/models/reshape2_mnist-8.onnx", "Input3")
    # ans[0] = result of inference, ans[1] = diagnostic output from modified layer, cf. onnxruntime_mod()
    print(ans[0])
    print_pretty(ans[1], 256, 10)

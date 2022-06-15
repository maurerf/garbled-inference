#!/usr/bin/env python3

import onnx, onnx.numpy_helper

model = onnx.load("mnist-8.onnx")
[tensor] = [t for t in model.graph.initializer if t.name == "Parameter194"]
w = onnx.numpy_helper.to_array(tensor)

print(w)

from __future__ import print_function
# import paddle.fluid as fluid
import paddle
import sys
import os
import numpy as np

# img = np.load("../dataset/MiniFasNet/input.npy")
# pytorch_output = np.load("../dataset/MiniFasNet/result.npy")
img = np.load("/home/shun/Documents/Projects/paddle/megemini/X2Paddle/test_benchmark/PyTorch/dataset/MiniFasNet/input.npy")
pytorch_output = np.load("/home/shun/Documents/Projects/paddle/megemini/X2Paddle/test_benchmark/PyTorch/dataset/MiniFasNet/result.npy")
f = open("result.txt", "w")
f.write("======MiniFasNet recognizer: \n")
try:
    # trace
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    # [prog, inputs, outputs] = fluid.io.load_inference_model(
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        # dirname="pd_model_trace/inference_model/",
        path_prefix="/home/shun/Documents/Projects/paddle/megemini/X2Paddle/test_benchmark/PyTorch/MiniFasNet/pd_model_trace/inference_model",
        executor=exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")
    result = exe.run(prog, feed={inputs[0]: img}, fetch_list=outputs)
    df = pytorch_output - result

    print(df)

    if np.max(np.fabs(df)) > 1e-03:
        print("Trace Failed", file=f)
    else:
        print("Trace Successed", file=f)
except:
    print("!!!!!Failed", file=f)

    raise

f.close()

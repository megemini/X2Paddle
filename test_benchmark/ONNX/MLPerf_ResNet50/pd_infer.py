import paddle
import numpy as np
import pickle
import sys

f = open('result.txt', 'w')
f.write("======MLPerf_ResNet50: \n")
try:
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())

    # test dygraph
    [prog, inputs, outputs] = paddle.static.load_inference_model(
        path_prefix="pd_model_dygraph/inference_model/model", executor=exe)
    data = np.load('../dataset/MLPerf_ResNet50/input_0.npy')
    result = exe.run(prog, feed={inputs[0]: data}, fetch_list=outputs)

    with open("../dataset/MLPerf_ResNet50/output_0.pkl", "rb") as fr:
        onnx_result = pickle.load(fr)
    is_successd = True
    for i in range(2):
        diff = result[i] - onnx_result[i]
        max_abs_diff = np.fabs(diff).max()
        if max_abs_diff >= 1e-05:
            relative_diff = max_abs_diff / np.fabs(onnx_result[i]).max()
            if relative_diff >= 1e-05:
                is_successd = False
    if is_successd:
        f.write("Dygraph Successed\n")
    else:
        f.write("!!!!!Dygraph Failed\n")
except:
    f.write("!!!!!Failed\n")

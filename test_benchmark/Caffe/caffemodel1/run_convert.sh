# 进行转换
x2paddle -f caffe -p deploy.prototxt -w ../dataset/caffemodel1/baidu_iter_250000.caffemodel -s pd_model_dygraph -df True
# 运行推理程序
python pd_infer.py

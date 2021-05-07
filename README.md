# 参考tensorflow官方教程完成的翻译机器人demo
作为练习备忘
参考：https://tensorflow.google.cn/tutorials/text/nmt_with_attention

# 工程结构介绍
（粗略介绍下）
model 模型文件夹
datasets 语料库
model.py 模型处理
train.py 封装模型训练和预测方法
utils.py 相关工具方法
start.py 启动文件 分别进行训练或者预测

# 初始运行说明
由于数据集和训练好的模型比较大，所以不会上传github。
第一次打开工程
首先新建model和datasets文件夹
然后运行start.py文件（更多的查看start.py说明）

# start.py说明
在这里进行训练启动或者预测。也预留了数据集下载的方法
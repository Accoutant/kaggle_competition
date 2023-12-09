# kaggle_competition
## data：储存数据处理的结果
比赛官方给的数据不存放在github中，复制仓库后请将官方给的数据解压到此处以确保其他文件可以正常运行。
## data_process：数据分析和数据处理过程文件
1、data_analysis_for_train.ipynb和train_and_weather.ipynb文件是特征分析文件。  
2、generate_data.ipynb文件用于生成train_data.pkl和train_data_selected.pkl文件,train_data.pkl文件是一个元组:(X, Y),  
X,Y都是nparray格式，分布储存特征和标签，train_data_selected.pkl是特征选择后的数据文件。
## logs：存放模型出图或者模型参数
## models: 模型文件
utils文件是各种公用函数
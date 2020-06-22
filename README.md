# recommandation-system
## 概述

一个训练非常非常慢的推荐系统大作业。

## 作业要求原文

Task: Predict the rating scores of the pairs `(u, i)` in the `Test.txt` file. （注：数据集在文件夹 `data-new/` 中，打分范围为 `[0,100]`）

Dataset： 

(1)  `Train.txt`, which is used for training your models. 

(2)  `Test.txt`, which is used for test. 

(3)  `ItemAttribute.txt`, which is used for training your models (optional). 

(4) `ResultForm.txt`, which is the form of your result file. 

The formats of datasets are explained in the DataFormatExplanation.txt. 

Note that if you can use `ItemAttribute.txt` appropriately and improve the performance of the algorithms, additional points (up to 10) can be added to your final course score. 

## 开发环境

Python 3.7.0 64-bit

## 功能描述

- [x] 使用基础带偏置的 SVD 算法作为基础推荐算法
- [x] 使用`ItemAttribute.txt`对推荐结果进行改进（使用最小二乘法，将物品属性作为输入，拟合训练集打分真实值和预测值之差）
- [x] 训练集和测试集的划分（随机数法，`Train.txt`中所有记录的 80% 作为训练集， 20% 作为测试集）
- [x] 对划分出的训练集进行 SVD 算法训练，并用物品属性拟合训练集打分真实值和最后一轮训练的预测值
- [x] 对测试集进行测试，并报告测试集 RMSE （在打分范围`[0,100]`的情况下，测试集 RMSE 在 `26.40` 左右）
- [x] 对 `Test.txt` 给定的记录进行预测，并输出结果文件 `result1.txt` （引入物品属性优化前）和 `result2.txt`（引入物品属性优化后）
- [ ] 使用并行化技术（或即时编译技术）对矩阵运算进行优化
  - 尝试使用 Python 的多进程和`numba`包的`jit`（即时编译）进行优化，效果微乎其微（或者根据无法运行），训练速度极慢

## 运行方式

直接运行 `recommand.py` 即可。

后面跟随的命令行参数格式如下：

```
[opt1] [arg1] [opt2] [arg2] ......
```

其中`opt`是选项，`arg`是这个选项对应的值。

命令行参数的选项和对应的值如下：

`-g`：重新生成训练集文件（`trainset.csv`）和测试集文件（`testset.csv`），默认是如果当前路径下有这两个文件则不重新生成，没有参数值

`-t`：重新进行训练，默认是如果当前路径下有训练结果文件（`p_matrix.dat`、`q_matrix.dat`、`bu_vector.dat`、`bi_vector.dat`、`sprase.dat`（稀疏矩阵文件））则不重新训练，没有参数值

`-h` / `--help`：显示帮助信息，没有参数值

`-m` / `--memory`：打印堆区内存详细信息（使用 `guppy` 库，对应 Python 3 版本为 `guppy3` ），没有参数值

`-f` / `--factors`：SVD 隐因子个数（默认为 `10` ）

`-e` / `--epochs`：SVD 训练时的迭代次数（默认为 `20` ）

`-r` / `--learnrate`：SVD 学习率（默认为 `0.005 `）

`-l` / `--lambda`：SVD 正则化参数（默认为 `0.02` ）

`-s` / `--scale`打分范围，格式为：`最低分,最高分`（英文逗号），默认值为：`0,100`

`--trainfile`：训练文件路径（默认为 `data-new/train.txt` ）

`--testfile`：测试文件路径（默认为 `data-new/test.txt` ）

`--attrfile`：item 属性文件路径（默认为 `data-new/itemAttribute.txt` ）
# 语音指令识别

实现了语音指令识别的任务，将left、right、stop作为识别词，go、on、off作为剩余词进行识别。

## 1. 目录结构

```
├── dataset            // 数据集
│   ├── README.md
│   ├── test           // 测试集，其中包括go，left，off，on，right，stop六个文件夹
│   └── train          // 训练集，其中包括go，left，off，on，right，stop六个文件夹
├── densenet           // 训练完成的densenet网络权重
├── densenet.png       // densenet网络结构图，使用keras.utils.plot_model绘制
├── densenet.py        // densenet网络实现代码
├── main.ipynb         // 训练主代码，详情见notebook中的注释
├── README.md
├── resnet50.py        // resnet网络实现代码
├── resnet.png         // resnet网络结构图，使用keras.utils.plot_model绘制
├── resnet_weight      // 训练完成的resnet网络权重
├── test.py            // 测试代码，使用集成的模型进行测试
├── tools.py           // 其他代码，包括读取数据、数据特征提取代码
├── vggnet19.py        // vgg网络实现代码
├── vgg_net_weight     // 训练完成的vgg网络权重
├── vgg.png            // vgg网络结构图，使用keras.utils.plot_model绘制
└── warmup_cosdecay.py // 余弦退火加warmup的代码
```

## 2. 运行方式

运行环境为ubuntu18.04。

### 2.1. 训练

进入jupyter依次运行数据读取、数据增强、提取mfcc特征、提取logmelspectrum特征四大块(具体代码分块情况见main.ipynb中的注释)，然后执行需要训练的网络的对应代码，完成训练。

### 2.2. 测试

使用测试脚本```test.py```，该脚本会读取```test_path```下的测试数据，进行特征提取、模型加载和集成并最终输入测试集上的预测准确率。注意**要求test_path文件夹下，将不同类别的语音放在对应名称为label的对应文件夹下，并且test_path要以左斜线结尾**，例如我的测试目录树是这样的：

```
dataset
├── README.md
├── test
│   ├── go
│   ├── left
│   ├── off
│   ├── on
│   ├── right
│   └── stop
└── train
    ├── go
    ├── left
    ├── off
    ├── on
    ├── right
    └── stop
```

对应测试命令如下：

```python3 test.py --test_path=dataset/test/```

输出结果为：

```accuracy: 0.9065743944636678```

## 3. 实验思路及结果

简要思路如下，详细思路见实验报告。

我选择将left、right、stop作为识别词，go、on、off作为剩余词进行分类。

代码提取了语音文件的mfcc特征和logmelspectrum特征，然后使用mfcc训练vgg19，使用logmelspectrum训练resnet和densenet网络，最终结果如下：

| 模型 | vgg | resnet | densenet | 集成三个模型 |
| ------ | ------ | ------ | ------ | ------ |
| 准确率 | 85.47 | 86.85 | 85.12 | 90.66 |

三个网络模型的效果基本差不多，集成后取得了很大的提升，提升的原因我认为主要是同时利用了mfcc和logmelspectrum两种特征。

## 4. TODO

只做了三天，有一些想法没有实现：首先是调整网络参数，多训练几次；其次是使用处理一维特征的模型，直接对声谱图处理，然后集成，具体参考[Kaggle上的一个解法](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47618)，同时使用三种不同的特征的话，应该是可以获得更好的效果的。

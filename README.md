<div align="center">
  <h1>猫狗分类项目</h1>
  <p>一个简单的基于PyTorch的图像分类项目，用于区分猫和狗。</p>
</div>

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---
## 📁 文件目录结构

以下是项目文件的目录结构：

```
猫狗分类项目/
├── task2.py           # 主程序，包含数据加载、模型定义、训练与测试逻辑
├── task2_1.py         # 实验过程中的阶段性脚本
├── task2_2.py         # 实验过程中的阶段性脚本
├── task2_3.py         # 实验过程中的阶段性脚本
├── requirements.txt   # Python 依赖列表
├── README.md          # 项目说明文件
├── LICENSE            # 许可证文件
├── cat_dog.zip        # 原始数据集（Kaggle Dogs vs. Cats），解压后得到 data/ 目录
└── data/              # 解压后的数据集目录
    ├── readbook/
    │   ├──Time Machine.txt
    ├── training_data/
    │   ├── cats/...
    │   └── dogs/...
    └── testing_data/
        ├── cats/...
        └── dogs/...
```

## 🚀 关于项目

这个项目是一个基本的图像分类任务，使用PyTorch来区分猫和狗的图像。它包括数据加载、模型定义、训练和测试。

### 主要功能

- **简单易懂**：非常适合深度学习的初学者。
- **模块化设计**：易于扩展或修改。
- **预训练模型**：包括基本的CNN和VGG风格的模型。

## 🛠️ 环境搭建

### 克隆仓库

```bash
git clone https://github.com/taboo-hacker/task2.git
cd task2
```

### 依赖要求

确保你已经安装了以下先决条件：

- Python
- torch
- torchsummary
- torchvision

你可以使用pip安装所有必需的包：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 数据准备

已经有了，你不需要管。

## 🏗️ 快速启动

1. 要开始训练模型，只需运行以下命令：

    ```bash
    python task2.py
    ```

2. 要测试模型，使用以下命令：
    
    ```bash
    python task2.py --test
    ```

## 🔍 模型架构

### CNN

一个简单的卷积神经网络，包含两个卷积层，后接一个dropout层和一个全连接层。

### VGG风格

一个更复杂的模型，灵感来自VGG架构，由五个卷积层组成，带有最大池化和dropout。

## 📊 结果

模型在训练50个周期后，在测试数据集上的准确率约为92%。

## 📄 许可证


本项目根据 [MIT License](LICENSE) 授权。详见 [LICENSE](LICENSE) 文件。

## 📧 联系方式

如有任何问题或建议，请随时联系：

- 邮箱：leo43991314520@outlook.com
- GitHub：[@taboo-hacker](https://github.com/taboo-hacker)

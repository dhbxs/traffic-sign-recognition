# 交通标志识别系统

本项目是一个基于深度学习的交通标志识别系统，使用卷积神经网络(CNN)对德国交通标志数据集(GTSRB)进行分类识别。
详细项目文档见: [https://blog.dhbxs.top/posts/4dCtXKfh](https://blog.dhbxs.top/posts/4dCtXKfh)

## 项目结构

```
.
├── data/                     # 数据集目录
│   ├── Train/               # 训练数据集（按类别分文件夹 0-42）
│   └── Test.csv             # 测试数据集索引文件
├── traffic_sign.py          # 主程序文件
├── traffic_classifier_me.h5 # 训练好的模型文件
└── log/                     # TensorBoard日志文件
```

## 环境依赖及版本

- Python 3.12
- TensorFlow/Keras
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow (PIL)

### 推荐安装命令

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn pillow
```

或者使用requirements.txt文件管理依赖：

```bash
pip install -r requirements.txt
```

## 数据集说明

项目使用德国交通标志识别基准数据集(GTSRB)，包含43类不同的交通标志：

- **训练数据**：位于 `data/Train/` 目录下，按类别分别存储在0-42编号的子文件夹中
- **测试数据**：`data/Test.csv` 文件包含测试图像路径和对应的真实标签

所有图像都会被预处理为30x30像素大小的彩色图像。

## 模型架构

采用卷积神经网络(CNN)架构，具体包括：

- 输入层：接收30x30x3的彩色图像
- 卷积层：多个Conv2D层用于特征提取
- 池化层：MaxPool2D层用于降维
- 正则化：Dropout层防止过拟合
- 全连接层：Dense层进行最终分类
- 输出层：43个节点的Softmax层输出各类别概率

## 如何运行项目

### 1. 环境准备

确保已安装所需依赖库：

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn pillow
```

### 2. 数据准备

- 将训练数据集按类别放入 `data/Train/` 目录下对应编号的文件夹中（0-42）
- 确保测试数据集信息在 `data/Test.csv` 文件中

### 3. 运行训练和测试

执行主程序文件：

```bash
python traffic_sign.py
```

程序将自动完成以下步骤：
1. 加载并预处理训练数据
2. 构建和编译CNN模型
3. 训练模型（默认11个epochs）
4. 保存训练好的模型为 `my_traffic_classifier.keras`
5. 在测试集上评估模型性能
6. 显示训练过程的准确率和损失图表

### 4. 查看训练过程

项目支持TensorBoard可视化：

```bash
tensorboard --logdir=log
```

然后在浏览器中访问 `http://localhost:6006` 查看详细训练过程。

## 模型参数

- **Epochs**: 11
- **Batch Size**: 32
- **优化器**: Adam
- **损失函数**: categorical_crossentropy
- **评估指标**: accuracy

## 输出结果

程序运行后将输出：
1. 训练过程中每个epoch的准确率和损失值图表
2. 测试集上的最终准确率

## 注意事项

1. 确保训练数据按照指定目录结构存放
2. 图像将被统一调整为30x30像素大小
3. 训练时间取决于硬件配置，建议使用GPU加速训练
4. 如果要重新训练模型，删除或备份原有的 `my_traffic_classifier.keras` 文件

## 可能的扩展方向

- 增加数据增强技术提高模型泛化能力
- 调整网络架构和超参数优化模型性能
- 添加实时摄像头识别功能
- 开发图形用户界面方便使用

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score
from keras.callbacks import TensorBoard

data = []
labels = []
classes = 43

# 当前路径windows版
cur_path = os.getcwd()

# log_path = os.getcwd() + "\\log"

# 当前路径mac版
log_path = os.getcwd() + "/log"

# 检索图像及其标签
for i in range(classes):
    path = os.path.join(cur_path, 'data/Train', str(i))
    images = os.listdir(path)

    for a in images:
        # print(a)
        # windows版
        # try:
        #     image = Image.open(path + '\\' + a)
        #     image = image.resize((30, 30))
        #     image = np.array(image)
        #     # sim = Image.fromarray(image)
        #     data.append(image)
        #     labels.append(i)
        # except:
        #     print("Error loading image")
        # mac版
        try:
            image = Image.open(path + '/' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            # sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# 将列表转换为numpy数组
data = np.array(data)
labels = np.array(labels)

# print(data.shape, labels.shape)
# 分割训练和测试数据集
# 训练集、测试集、训练标签集、测试标签集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 将标签转换为一种热编码(将数据扩维)One-Hot编码
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
# print(y_test)

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
# model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.25))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.25))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(43, activation='softmax'))

# 建立模型
model = Sequential()
# 添加卷积输入层 16个节点 5*5的卷积核大小
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))

# 卷积层 + 最大池化层
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
# 防止过拟合，网络正则化，随机消灭上一层的神经元
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# 展平层
model.add(Flatten())
# 密集连接层
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
# 全连接 + 输出层
model.add(Dense(43, activation='softmax'))

# 编译模型 分类交叉熵损失函数 Adam优化器 这种搭配常用在多元分类中
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 11
tensorboard = TensorBoard(log_dir='./log', histogram_freq=1, write_graph=True, write_images=True, update_freq="epoch")

history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test), callbacks=[tensorboard])
model.save("traffic_classifier_me.h5")

# 绘制图形以确保准确性
plt.figure(0)
# 训练集准确率
plt.plot(history.history['accuracy'], label='training accuracy')
# 测试集准确率
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# 测试数据集的测试准确性

y_test = pd.read_csv('data/Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)

pred = model.predict_classes(X_test)

# 测试数据的准确性
print(accuracy_score(labels, pred))

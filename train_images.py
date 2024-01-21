

import numpy as np
import matplotlib.pyplot as plt
# 定义辅助函数加载MNIST数据集
import gzip


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)  # 图像尺寸为28x28
    return data


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data
# 加载训练集图像和标签
train_images = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')

# 随机选择几个图像
num_images = 5
random_indices = np.random.choice(len(train_images), num_images, replace=False)
selected_images = train_images[random_indices]
selected_labels = train_labels[random_indices]

# 绘制图像
fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
for i, (image, label) in enumerate(zip(selected_images, selected_labels)):
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import shuffle

    # 定义辅助函数加载MNIST数据集
    import gzip
    # 加载plot库用于Debug
    import matplotlib.pyplot as plt

    import numpy as np


    # 读取图像文件
    def load_images(filename):
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)  # 图像尺寸为28x28
        return data


    # 读取标签文件
    def load_labels(filename):
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data


    # 定义文件路径
    train_images_file = 'train-images-idx3-ubyte'
    train_labels_file = 'train-labels-idx1-ubyte'
    test_images_file = 't10k-images-idx3-ubyte'
    test_labels_file = 't10k-labels-idx1-ubyte'

    # 加载训练集图像和标签
    train_images = load_images(train_images_file)
    train_labels = load_labels(train_labels_file)

    # 加载测试集图像和标签
    test_images = load_images(test_images_file)
    test_labels = load_labels(test_labels_file)


    # 数据预处理
    X_train, y_train = train_images.reshape(len(train_images), -1), train_labels
    X_test, y_test = test_images.reshape(len(test_images), -1), test_labels

    # 特征缩放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # 构建模型
    class NeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size, regularization_strength, learning_rate):
            self.W1 = np.random.randn(input_size, hidden_size)
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size)
            self.b2 = np.zeros(output_size)
            self.regularization_strength = regularization_strength
            self.learning_rate = learning_rate

        def forward(self, X):
            self.z1 = np.dot(X, self.W1) + self.b1
            # Z1标准化
            self.z1=(self.z1-np.mean(self.z1))/np.var(self.z1)
            self.a1 = np.maximum(0, self.z1)
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            # probs softmax
            exp_scores = np.exp(self.z2)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return self.probs

        def backward(self, X, y):
            num_examples = X.shape[0]
            delta3 = self.probs
            delta3[range(num_examples), y] -= 1
            dW2 = np.dot(self.a1.T, delta3)
            db2 = np.sum(delta3, axis=0)
            delta2 = np.dot(delta3, self.W2.T)
            # delta2 逆向求的的第一层输出与目标的差值
            delta2[self.a1 <= 0] = 0
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)
            # 加入L2正则化项的梯度更新
            dW2 += self.regularization_strength * self.W2
            dW1 += self.regularization_strength * self.W1
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

        def train(self, X, y, num_epochs):
            for epoch in range(num_epochs):
                self.forward(X)
                self.backward(X, y)

        def predict(self, X):
            return np.argmax(self.forward(X), axis=1)


    # 设置超参数
    input_size = X_train.shape[1]
    hidden_size = 1024
    output_size = len(np.unique(y_train))
    learning_rate = 0.00001
    num_epochs = 20
    regularization_strength = 0.01  # L2正则化参数


    # 创建神经网络对象并测试多个epochs
    epochs_to_test = range(1,31,5)  # 要测试的epochs列表
    test_accuracies = []  # 存储测试集准确度
    train_accuracies = []  # 存储训练集准确度
    for num_epochs in epochs_to_test:
        model = NeuralNetwork(input_size, hidden_size, output_size, regularization_strength, learning_rate)
        model.train(X_train, y_train, num_epochs)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        test_accuracies.append(accuracy)
        print(f"Epochs: {num_epochs} - Test accuracy: {accuracy}")

        y_pred_train = model.predict(X_train)
        train_accuracy = np.mean(y_pred_train == y_train)
        train_accuracies.append(train_accuracy)
        print(f"Epochs: {num_epochs} - Train accuracy: {train_accuracy}")

    # 绘制准确度-epochs曲线
    import matplotlib.pyplot as plt

    plt.plot(epochs_to_test, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_to_test, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy vs. Epochs')
    plt.legend()
    plt.show()
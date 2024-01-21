if __name__ == '__main__':
    # Loading the numpy library for math operations
    import numpy as np
    # Loading StandardScaler for standardization
    from sklearn.preprocessing import StandardScaler
    # Loading the matplotlib library for presenting test results
    import matplotlib.pyplot as plt

    # Reading image files
    def load_images(filename):
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)  # image size 28x28
        return data

    # Read label file
    def load_labels(filename):
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    # Define the file path
    train_images_file = 'train-images-idx3-ubyte'
    train_labels_file = 'train-labels-idx1-ubyte'
    test_images_file = 't10k-images-idx3-ubyte'
    test_labels_file = 't10k-labels-idx1-ubyte'

    # Load training set images and labels
    train_images = load_images(train_images_file)
    train_labels = load_labels(train_labels_file)

    # Load test set images and labels
    test_images = load_images(test_images_file)
    test_labels = load_labels(test_labels_file)

    # Data pre-processing
    X_train, y_train = train_images.reshape(len(train_images), -1), train_labels
    X_test, y_test = test_images.reshape(len(test_images), -1), test_labels

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Build NeuralNetwork class
    class NeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size, regularization_strength, learning_rate):
            self.W1 = np.random.randn(input_size, hidden_size)
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size)
            self.b2 = np.zeros(output_size)
            self.regularization_strength = regularization_strength
            self.learning_rate = learning_rate

        def forward(self, X):
            self.z1 = np.dot(X, self.W1) + self.b1                              # Z1 = Output of input layer
            self.z1=(self.z1-np.mean(self.z1))/np.var(self.z1)                  # Standardize Z1
            self.a1 = np.maximum(0, self.z1)                                    # ReLU Activation Functions => Non-Linear
            self.z2 = np.dot(self.a1, self.W2) + self.b2                        # Z2 = Output of hidden layer
            exp_scores = np.exp(self.z2)                                        # Z2 softmax(1)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # Z2 softmax(2)
            return self.probs

        def backward(self, X, y):
            num_examples = X.shape[0]             # number of image in this backward process
            delta3 = self.probs                   # Prediction of output
            delta3[range(num_examples), y] -= 1   # Back Propagation to obtain ∂L/∂z
            dW2 = np.dot(self.a1.T, delta3)       # Back Propagation to obtain dw = ∑(y2'-y2) * a1
            db2 = np.sum(delta3, axis=0)          # Back Propagation to obtain db = ∑(y2'-y2)

            # delta2 is the difference between the output of
            # the hidden layer of the inverse solution and the target
            delta2 = np.dot(delta3, self.W2.T)    # computing the error propagated from the output layer to the hidden layer›
            delta2[self.a1 <= 0] = 0              # Applying ReLU activation derivative
            dW1 = np.dot(X.T, delta2)             # Backpropagation to obtain dw = ∑(y1'-y1) * X
            db1 = np.sum(delta2, axis=0)          # Backpropagation to obtain db = ∑(y1'-y1)

            # Gradient update with the addition of the L2 regularization term
            dW2 += self.regularization_strength * self.W2
            dW1 += self.regularization_strength * self.W1

            self.W1 -= self.learning_rate * dW1  # Updating weights of the first layer
            self.b1 -= self.learning_rate * db1  # Updating biases of the first layer
            self.W2 -= self.learning_rate * dW2  # Updating weights of the second layer
            self.b2 -= self.learning_rate * db2  # Updating biases of the second layer

        def train(self, X, y, num_epochs):
            for epoch in range(num_epochs):
                self.forward(X)
                self.backward(X, y)

        def predict(self, X):
            return np.argmax(self.forward(X), axis=1)


    # Setting hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 1024
    output_size = len(np.unique(y_train))
    learning_rate = 0.00001
    num_epochs = 20
    regularization_strength = 0.01  # The L2 regularization parameter

    # Test Program
    hidden_sizes = [256, 512, 1024, 2048]  # List of hidden layer sizes to test
    epochs_to_test = range(1, 31, 5)  # List of epochs to test

    for hidden_size in hidden_sizes:
        test_accuracies = []
        train_accuracies = []

        for num_epochs in epochs_to_test:
            model = NeuralNetwork(input_size, hidden_size, output_size, regularization_strength, learning_rate)
            model.train(X_train, y_train, num_epochs)

            y_pred = model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            test_accuracies.append(accuracy)
            print(f"Hidden Size: {hidden_size}, Epochs: {num_epochs} - Test accuracy: {accuracy}")

        plt.plot(epochs_to_test, test_accuracies, label=f'Test Accuracy (Hidden Size: {hidden_size})')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs. Epochs')
    plt.legend()
    plt.show()
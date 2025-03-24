import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        log_probs = -np.log(A2[range(m), Y])
        return np.sum(log_probs) / m
    
    def backward_pass(self, X, Y):
        m = X.shape[0]
        dZ2 = self.A2
        dZ2[range(m), Y] -= 1
        dZ2 /= m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            A2 = self.forward_pass(X)
            loss = self.compute_loss(Y, A2)
            self.backward_pass(X, Y)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict(self, X):
        A2 = self.forward_pass(X)
        return np.argmax(A2, axis=1)

if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.rand(1000, 784)  # 1000 samples of 28x28 images flattened
    Y_train = np.random.randint(0, 10, 1000)  # Random labels (0-9)
    
    nn = SimpleNeuralNetwork(784, 128, 10, learning_rate=0.1)
    nn.train(X_train, Y_train, epochs=100000)

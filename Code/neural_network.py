import numpy as np
from sklearn.preprocessing import OneHotEncoder
class NeuralNetwork:
    
    def __init__(self, input_dim, hidden_dim, output_dim, l1_lambda):
        np.random.seed(1)
        W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        b1 = np.zeros((1, hidden_dim))
        W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        b2 = np.zeros((1, output_dim))
        self.weights = [W1, W2]
        self.biases = [b1, b2]
        self.l1_lambda = l1_lambda
    
    #ReLU activation function
    def relu(self, x):
        return np.maximum(0, x)

    # Derivative of ReLU
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    #Softmax activation function
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        self.h_input = np.dot(X, self.weights[0]) + self.biases[0]
        self.h_output = self.relu(self.h_input)
        self.o_input = np.dot(self.h_output, self.weights[1]) + self.biases[1]
        self.prediction = self.softmax(self.o_input)
        return  self.prediction

    def backward_propagation(self, X, y_true):
        num_samples = X.shape[0]
        
        output_layer_error = self.prediction - y_true 
        dW_output = np.dot(self.h_output.T, output_layer_error) / num_samples + self.l1_regularization(self.weights[1])
        db_output = np.sum(output_layer_error, axis=0) / num_samples
        
        hidden_layer_error = np.dot(output_layer_error, self.weights[1].T)  
        d_hidden_input = hidden_layer_error * self.relu_derivative(self.h_input) 
        dW_hidden = np.dot(X.T, d_hidden_input) / num_samples + self.l1_regularization(self.weights[0])
        db_hidden = np.sum(d_hidden_input, axis=0) / num_samples
        
        return dW_hidden, db_hidden, dW_output, db_output

    def l1_regularization(self, weights):
        return self.l1_lambda * np.sign(weights)
    
    
    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)
    
    def loss_function(self, X, y_true):
        y_pred = self.forward_propagation(X)
        
        # Compute cross-entropy loss
        epsilon = 1e-15  # To avoid log(0) issues
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cross_entropy_loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        # Compute L1 regularization loss
        regularization_loss = self.l1_lambda * (np.sum(np.abs(self.weights[0])) + np.sum(np.abs(self.weights[1])))

        return cross_entropy_loss + regularization_loss

    def update_parameters(self, dW_hidden, db_hidden, dW_output, db_output):
            self.weights[0] -= self.learning_rate * dW_hidden
            self.biases[0] -= self.learning_rate * db_hidden
            self.weights[1] -= self.learning_rate * dW_output
            self.biases[1] -= self.learning_rate * db_output


    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
            self.learning_rate = learning_rate
            for epoch in range(epochs):
                self.forward_propagation(X_train)
                
                dW_hidden, db_hidden, dW_output, db_output = self.backward_propagation(X_train, y_train)
                
                self.update_parameters(dW_hidden, db_hidden, dW_output, db_output)
                
                # Compute loss
                if epoch % 100 == 0:
                    train_loss = self.loss_function(X_train, y_train)
                    val_loss = self.loss_function(X_val, y_val)
                    print(f'Epoch {epoch}: Train Loss = {train_loss}, Validation Loss = {val_loss}')

                # Optionally, you can track the training progress (e.g., using plt.plot)
            print('Training complete!')
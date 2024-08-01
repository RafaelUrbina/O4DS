import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork 
from utils import load_mnist

X_train, y_train, X_test, y_test = load_mnist()
    
    # Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    
    # Initialize and train the neural network
nn = NeuralNetwork(input_dim=28*28, hidden_dim=128, output_dim=10, l1_lambda=0.01)
nn.train(X_train, y_train, X_val, y_val, epochs=1000, learning_rate=0.01)
    
    # Predict on test data and compute accuracy
y_pred = nn.predict(X_test)
y_true = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_true, y_pred)
    
print(f'Test Accuracy: {accuracy * 100:.2f}%')
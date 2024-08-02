import numpy as np

class HeavyBallOptimizer:
    def __init__(self, model, momentum):
        self.model = model
        self.learning_rate = model.learning_rate
        self.momentum = momentum
        self.v_W1 = np.zeros_like(model.W1)
        self.v_b1 = np.zeros_like(model.b1)
        self.v_W2 = np.zeros_like(model.W2)
        self.v_b2 = np.zeros_like(model.b2)

    def update(self, model, momentum):
        self.__init__(model, momentum)
        self.v_W1 = self.momentum * self.v_W1 - self.learning_rate * self.model.dW_hidden
        self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * self.model.db_hidden
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * self.model.dW_output
        self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * self.model.db_output

        self.model.W1 += self.v_W1
        self.model.b1 += self.v_b1
        self.model.W2 += self.v_W2
        self.model.b2 += self.v_b2

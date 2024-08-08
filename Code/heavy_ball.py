import numpy as np

class HeavyBallOptimizer:
    def __init__(self, model, momentum, max_iter=1000, grad_tol=1e-12):
        self.model = model
        self.learning_rate = model. learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.grad_tol = grad_tol
        self.v_W1 = np.zeros_like(model.W1)
        self.v_b1 = np.zeros_like(model.b1)
        self.v_W2 = np.zeros_like(model.W2)
        self.v_b2 = np.zeros_like(model.b2)

    def norm(self, gradients):
        return np.linalg.norm(np.concatenate([g.flatten() for g in gradients]))

    def optimize(self, X, y, outlist=True):
        loss_history = []
        best_loss = []

        for i in range(self.max_iter):
            current_loss = self.model.evaluate(X, y)

            if outlist:
                loss_history.append(current_loss)
                if len(best_loss) > 0:
                    best_loss.append(min(best_loss[-1], current_loss))
                else:
                    best_loss.append(current_loss)

            g_W1, g_b1, g_W2, g_b2 = self.model.gradient(X, y)
            gradients = [g_W1, g_b1, g_W2, g_b2]
            grad_norm = self.norm(gradients)

            # Check stopping conditions
            if grad_norm <= self.grad_tol:
                break

            # Update velocity and parameters
            self.v_W1 = self.momentum * self.v_W1 - self.learning_rate * g_W1
            self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * g_b1
            self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * g_W2
            self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * g_b2

            self.model.W1 += self.v_W1
            self.model.b1 += self.v_b1
            self.model.W2 += self.v_W2
            self.model.b2 += self.v_b2

        if outlist:
            return loss_history, best_loss

        return self.model.W1, self.model.b1, self.model.W2, self.model.b2

# Example usage:
# optimizer = HeavyBallOptimizer(model, momentum=0.9)
# loss_history, best_loss = optimizer.optimize(X_train, y_train)

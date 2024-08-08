import numpy as np
from cvxopt import matrix, solvers

class BundleMethod:
    def __init__(self,model, max_bundle_size, max_iter):
        self.model = model
        self.max_bundle_size = max_bundle_size
        self.bundle = []
        self.max_iter = max_iter

    def update(self, model, bundle_size):
        self.__init__(model, bundle_size)
        
        for i in range(self.max_iter):
            # Compute the subgradient (use the gradient from backpropagation)
            subgradient = [self.model.dW_hidden, self.model.dW_output]
            # Compute the function value
            function_value = self.model.loss_function(self.X_train, self.y_train)
            
            # Add the new subgradient and function value to the bundle
            if len(self.bundle) == self.max_bundle_size:
                self.bundle.pop(0)  # Remove the oldest element if bundle size exceeds the max limit

            self.bundle.append((subgradient, function_value))

            # Solve the master problem using cvxopt
            self.solve_master_problem()

    def solve_master_problem(self):
        m = len(self.bundle)
        if m == 0:
            return

        # Create the matrices for the QP problem
        P = matrix(np.eye(m))
        q = matrix(np.zeros(m))
        G = matrix(np.vstack([-np.eye(m), np.zeros((1, m))]))
        h = matrix(np.hstack([np.zeros(m), np.array([1])]))

        # Construct the objective function (c)
        c = np.array([fv for _, fv in self.bundle])

        # Formulate the problem
        P = 2 * matrix(np.dot(c[:, None], c[None, :]))
        q = matrix(-c)

        # Solve the QP problem
        sol = solvers.qp(P, q, G, h)
        alphas = np.array(sol['x']).flatten()

        # Use the solution to update weights
        avg_subgradient = [np.zeros_like(g) for g in self.bundle[0][0]]
        for alpha, (sg, _) in zip(alphas, self.bundle):
            avg_subgradient[0] += alpha * sg[0]
            avg_subgradient[1] += alpha * sg[1]

        self.model.weights[0] -= self.model.learning_rate * avg_subgradient[0]
        self.model.weights[1] -= self.model.learning_rate * avg_subgradient[1]
        self.model.biases[0] -= self.model.learning_rate * self.model.db_hidden
        self.model.biases[1] -= self.model.learning_rate * self.model.db_output

import numpy as np

class MiniBatchGradientDescent:
    def __init__(self, learning_rate=0.01, batch_size=32, epochs=100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def compute_gradient(self, X, y, weights):
        """
        Computes the gradient of the cost function with respect to the weights.
        :param X: Input data
        :param y: Actual labels
        :param weights: Current weights of the model
        :return: Gradient of the weights
        """
        m = len(y)
        predictions = X.dot(weights)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        return gradient

    def update_weights(self, weights, gradient):
        """
        Updates the weights using gradient descent.
        :param weights: Current weights of the model
        :param gradient: Computed gradient
        :return: Updated weights
        """
        return weights - self.learning_rate * gradient

    def fit(self, X, y):
        """
        Fit the model using mini-batch gradient descent.
        :param X: Input features
        :param y: Target values
        :return: Trained weights
        """
        m, n = X.shape
        weights = np.zeros(n)
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Compute gradient and update weights
                gradient = self.compute_gradient(X_batch, y_batch, weights)
                weights = self.update_weights(weights, gradient)

        return weights

    def predict(self, X, weights):
        """
        Make predictions using the learned weights.
        :param X: Input features
        :param weights: Trained weights of the model
        :return: Predicted values
        """
        return X.dot(weights)

# Example usage:
if __name__ == "__main__":
    # Generate some random data for demonstration
    np.random.seed(42)
    X = np.random.rand(1000, 3)  # 1000 samples, 3 features
    y = X.dot(np.array([3, 5, 2])) + np.random.randn(1000) * 0.5  # true weights [3, 5, 2]

    # Initialize mini-batch gradient descent
    mbgd = MiniBatchGradientDescent(learning_rate=0.01, batch_size=64, epochs=100)

    # Train the model
    weights = mbgd.fit(X, y)
    
    # Make predictions
    predictions = mbgd.predict(X, weights)

    print(f"Trained weights: {weights}")

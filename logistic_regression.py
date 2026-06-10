import numpy as np
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iters=1000, lambda_param=0.01, batch_size=32):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    def _compute_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        l2_penalty = (self.lambda_param / 2) * np.sum(self.weights ** 2)
        return log_loss + l2_penalty
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iters):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]
                curr_size = X_batch.shape[0]
                linear = X_batch @ self.weights + self.bias
                y_pred = self._sigmoid(linear)
                error = y_pred - y_batch
                dw = (X_batch.T @ error) / curr_size + (self.lambda_param * self.weights)
                db = np.sum(error) / curr_size
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            full_preds = self._sigmoid(X @ self.weights + self.bias)
            self.loss_history.append(self._compute_loss(y, full_preds))
        return self
    def predict(self, X, threshold=0.5):
        return (self._sigmoid(X @ self.weights + self.bias) >= threshold).astype(int)

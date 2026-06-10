import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score
from logistic_regression import LogisticRegressionScratch

np.random.seed(42)
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_samples = X_train.shape[0]
model = LogisticRegressionScratch(learning_rate=0.1, n_iters=100, lambda_param=1/n_samples)
model.fit(X_train, y_train)
sk_model = SklearnLR(penalty='l2', C=1.0, max_iter=1000)
sk_model.fit(X_train, y_train)
print(f"Scratch Accuracy:   {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(f"Sklearn Accuracy:   {accuracy_score(y_test, sk_model.predict(X_test)):.4f}")
print(f"Weight Correlation: {np.corrcoef(model.weights, sk_model.coef_.ravel())[0,1]:.4f}")
print(f"Final train loss:   {model.loss_history[-1]:.4f}")
print(f"First train loss:   {model.loss_history[0]:.4f}")
print(f"Bias scratch/sk:    {model.bias:.4f} / {sk_model.intercept_[0]:.4f}")

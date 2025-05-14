import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------------
# 1. Load the Dataset
# -------------------------------
path = "C:/Users/Ansar-PC/Desktop/developerhub-data-analysis/BostonHousing.csv"
df = pd.read_csv(path)

# -------------------------------
# 2. Data Preprocessing
# -------------------------------
# Separate features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Normalize numerical features
X_norm = (X - X.mean()) / X.std()

# Split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm.values, y.values, test_size=0.2, random_state=42)

# -------------------------------
# 3. Linear Regression (From Scratch)
# -------------------------------
class LinearRegressionScratch:
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef_

# -------------------------------
# 4. Random Forest Regressor (Simplified)
# -------------------------------
from collections import Counter
import random

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def fit(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.best_split = self._best_split(X, y)
        if self.best_split is None or depth >= self.max_depth:
            self.prediction = np.mean(y)
            return
        
        left_idx, right_idx = self.best_split["indices"]
        self.left = DecisionTreeRegressor(self.max_depth)
        self.left.fit(X[left_idx], y[left_idx], depth + 1)
        self.right = DecisionTreeRegressor(self.max_depth)
        self.right.fit(X[right_idx], y[right_idx], depth + 1)

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_split = None
        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = np.where(X[:, feature] <= threshold)[0]
                right_idx = np.where(X[:, feature] > threshold)[0]
                if len(left_idx) < self.min_samples_split or len(right_idx) < self.min_samples_split:
                    continue
                left_mse = np.var(y[left_idx]) * len(left_idx)
                right_mse = np.var(y[right_idx]) * len(right_idx)
                mse = (left_mse + right_mse) / self.n_samples
                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "indices": (left_idx, right_idx)
                    }
        return best_split

    def predict_single(self, x):
        if hasattr(self, "prediction"):
            return self.prediction
        if x[self.best_split["feature"]] <= self.best_split["threshold"]:
            return self.left.predict_single(x)
        else:
            return self.right.predict_single(x)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

class RandomForestRegressorScratch:
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            idxs = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return tree_preds.mean(axis=0)

# -------------------------------
# 5. XGBoost Regressor (Simplified)
# -------------------------------
class XGBoostScratch:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.gains = []

    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)
            self.gains.append(update)

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for update in self.trees:
            pred += self.learning_rate * update.predict(X)
        return pred

# -------------------------------
# 6. Performance Metrics
# -------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_residual / ss_total

# -------------------------------
# 7. Train and Evaluate Models
# -------------------------------
models = {
    "Linear Regression": LinearRegressionScratch(),
    "Random Forest": RandomForestRegressorScratch(n_estimators=10, max_depth=5),
    "XGBoost": XGBoostScratch(n_estimators=10, learning_rate=0.1)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "RMSE": rmse(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

print("\nModel Performance Comparison:")
for name, metrics in results.items():
    print(f"{name}: RMSE = {metrics['RMSE']:.2f}, R² = {metrics['R2']:.2f}")

# -------------------------------
# 8. Feature Importance (Random Forest)
# -------------------------------
def compute_feature_importance(forest):
    importance = defaultdict(float)
    for tree in forest.trees:
        if hasattr(tree, 'best_split'):
            def traverse(node):
                if hasattr(node, 'best_split'):
                    feature = node.best_split['feature']
                    importance[feature] += 1
                    traverse(node.left)
                    traverse(node.right)
            traverse(tree)
    return importance

rf_importance = compute_feature_importance(models["Random Forest"])
sorted_feat = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)

features = [X.columns[i] for i, _ in sorted_feat]
scores = [score for _, score in sorted_feat]

plt.figure(figsize=(10, 6))
plt.barh(features, scores)
plt.xlabel("Feature Importance (Count)")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

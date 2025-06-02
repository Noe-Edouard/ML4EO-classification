from sklearn.ensemble import RandomForestClassifier

class RandomForestSegmentation:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y):
        N, H, W, C = X.shape
        X_flat = X.reshape(-1, C)
        y_flat = y.reshape(-1)
        self.model.fit(X_flat, y_flat)

    def predict(self, X):
        N, H, W, C = X.shape
        X_flat = X.reshape(-1, C)
        preds_flat = self.model.predict(X_flat)
        return preds_flat.reshape(N, H, W)

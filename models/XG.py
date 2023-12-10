import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import pickle
import matplotlib as plt

with open("../data/train_data_selected.pkl", "rb") as f:
    X, Y = pickle.load(f)


def make_teain_test(X, Y, seed, rate):
    idx = int(rate * X.shape[0])
    X_train = X[:idx]
    Y_train = Y[:idx]
    X_test = X[idx:]
    Y_test = Y[idx:]
    shuffled_indices = np.arange(X_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffled_indices)
    X_train, Y_train = X_train[shuffled_indices], Y_train[shuffled_indices]
    return (X_train, Y_train), (X_test, Y_test)

(X_train, y_train), (X_test, y_test) = make_teain_test(X, Y, 1, 0.7)


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
clf = xgb.XGBRegressor(
                        device = device,
                        objective='reg:absoluteerror',
                        n_estimators = 2 if False else 1500,
                        early_stopping_rounds=100
                       )
clf.fit(X = X_train,
        y = y_train,
        eval_set = [(X_train, y_train), (X_test, y_test)],
        verbose = True
       )
print(clf.predict(X_test))
print(f'Early stopping on best iteration #{clf.best_iteration} with MAE error on validation set of {clf.best_score:.2f}')
# results = clf.evals_result()
# train_mae, val_mae = results["validation_0"]["mae"], results["validation_1"]["mae"]
# x_values = range(0, len(train_mae))
# fig, ax = plt.subplots(figsize=(8,4))
# ax.plot(x_values, train_mae, label="Train MAE")
# ax.plot(x_values, val_mae, label="Validation MAE")
# ax.legend()
# plt.ylabel("MAE Loss")
# plt.title("XGBoost MAE Loss")
# plt.show()


import lightgbm as lgb
import pickle
from utils import make_train_test
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

with open('../data/train_data.pkl', 'rb') as f:
    X, Y = pickle.load(f)

(X_train, y_train), (X_test, y_test) = make_train_test(X, Y, 1, 0.7)


params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression', # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31,   # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # &lt;0 显示致命的, =0 显示错误 (警告), &gt;0 显示信息
}

model = lgb.LGBMRegressor(objective='regression', n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(mean_absolute_error(y_pred, y_test))

# 画特征图
lgb.plot_importance(model, max_num_features=20)
plt.show()
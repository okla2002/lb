# coding: utf-8

# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# 指定字体以避免空白格问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据加载和可视化
# 加载四组锂电池数据，分别为CS2_35.csv, CS2_36.csv, CS2_37.csv, CS2_38.csv
df_35 = pd.read_csv('CS2_35.csv', index_col=0).dropna()
df_36 = pd.read_csv('CS2_36.csv', index_col=0).dropna()
df_37 = pd.read_csv('CS2_37.csv', index_col=0).dropna()
df_38 = pd.read_csv('CS2_38.csv', index_col=0).dropna()

# 绘制四种属性的变化图
fig = plt.figure(figsize=(9, 8), dpi=150)
names = ['capacity', 'resistance', 'CCCT', 'CVCT']
titles = ['放电容量 (mAh)', '内阻 (Ohm)', '恒流充电时间 (s)', '恒压充电时间 (s)']
plt.subplots_adjust(hspace=0.25)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(df_35[names[i]], 'o', ms=2, label='#35')
    plt.plot(df_36[names[i]], 'o', ms=2, label='#36')
    plt.plot(df_37[names[i]], 'o', ms=2, label='#37')
    plt.plot(df_38[names[i]], 'o', ms=2, label='#38')
    plt.title(titles[i], fontsize=14)
    plt.legend(loc='upper right')
    if i == 3:
        plt.ylim(1000, 5000)
    plt.xlim(-20, 1100)
plt.show()

# 数据处理和准备
def ConvertData(dataset, t_width):
    X_trains = []
    y_trains = []

    for df in dataset:
        t_length = len(df)
        train_x = np.arange(t_length)
        capacity = np.array(df['capacity'])
        train_y = capacity

        for i in range(t_length - t_width):
            X_trains.append(train_y[i:i + t_width])
            y_trains.append(train_y[i + t_width])

    X_trains = np.array(X_trains)
    y_trains = np.array(y_trains)

    return X_trains, y_trains

# 将数据转换为模型可用的格式
X_train, y_train = ConvertData([df_35, df_37, df_38], 50)
X_test, y_test = ConvertData([df_36], 50)

# 输出数据形状确认
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 随机选择500条数据进行训练
idx = np.random.permutation(np.arange(0, X_train.shape[0], 1))[:500]
X_train = X_train[idx]
y_train = y_train[idx]

# 数据形状调整
X_train = X_train.reshape([X_train.shape[0], X_train.shape[1]])
X_test = X_test.reshape([X_test.shape[0], X_test.shape[1]])
y_train = y_train.reshape([y_train.shape[0]])
y_test = y_test.reshape([y_test.shape[0]])
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 模型建立
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 训练集上的预测结果
predicted_train = rf_model.predict(X_train)

# 可视化训练集预测结果
fig = plt.figure(figsize=(12, 4), dpi=150)
plt.plot(predicted_train, alpha=0.7, label='预测值')
plt.plot(y_train, alpha=0.7, label='真实值')
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('数据点', fontsize=13)
plt.ylabel('放电容量 (Ah)', fontsize=13)
plt.title('随机森林模型预测训练数据的放电容量', fontsize=14)
plt.show()

# 测试集上的预测结果
predicted_test = rf_model.predict(X_test[300:800])

# 可视化测试集预测结果
x_range = np.linspace(301, 800, 500)
fig = plt.figure(figsize=(12, 4), dpi=150)
plt.plot(x_range, predicted_test, label='预测值')
plt.plot(x_range, y_test[300:800], label='真实值')
plt.xlabel('循环次数', fontsize=13)
plt.ylabel('放电容量 (Ah)', fontsize=13)
plt.title('随机森林模型预测测试数据（CS2-36）的放电容量', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.show()

# 计算测试集预测结果的均方误差
mse = mean_squared_error(y_test[300:800], predicted_test)
print("均方误差 (MSE)：", mse)

# 未来数据点的预测
initial = X_test[500]
results = []
for i in tqdm(range(50), ascii=True):  # 添加ascii=True参数
    if i == 0:
        res = rf_model.predict(initial.reshape(1, -1))
        results.append(res[0])
    else:
        initial = np.append(initial[1:], res)
        res = rf_model.predict(initial.reshape(1, -1))
        results.append(res[0])

# 可视化未来数据点的预测结果
fig = plt.figure(figsize=(12, 4), dpi=150)
plt.plot(np.linspace(501, 550, 50), results, 'o-', ms=4, lw=1, label='预测值')
plt.plot(np.linspace(401, 550, 150), y_test[400:550], 'o-', lw=1, ms=4, label='真实值')
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('循环次数', fontsize=13)
plt.ylabel('放电容量 (Ah)', fontsize=13)
plt.title('未来50个循环的放电容量预测', fontsize=14)
plt.show()
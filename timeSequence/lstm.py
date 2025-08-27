import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 数据预处理
def create_sequences(data, time_steps):
    """
    创建用于LSTM训练的序列数据
    
    参数:
    data: 一维数组，时间序列数据
    time_steps: 时间窗口大小，用过去time_steps个时间点预测下一个时间点
    
    返回:
    X: 特征序列 (samples, time_steps, features)
    y: 目标值 (samples,)
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# 构建LSTM模型
def build_lstm_model(input_shape, n_feature, units=[50, 50], dropout_rate=0.2):
    """
    构建LSTM模型，假设只有一个要预测的 feature
    
    参数:
    input_shape: 输入数据形状 (time_steps, features)
    units: 每层LSTM的神经元数量
    dropout_rate: Dropout比率，防止过拟合
    
    返回:
    model: 编译好的Keras模型
    """
    model = Sequential()
    
    # 第一层LSTM
    model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # # 第二层LSTM
    model.add(LSTM(units[1], return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # 输出层
    model.add(Dense(n_feature))
    
    # 编译模型
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def plot_history(history):
    """
    绘制训练过程中的损失和评估指标变化

    参数:
    history: 训练过程的历史记录
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 未来预测
def predict_future(model, scaled_data, future_steps, scaler, time_steps):
    """
    使用训练好的模型预测未来多个时间步
    
    参数:
    model: 训练好的LSTM模型
    time_steps: 使用的时间序列窗口
    future_steps: 要预测的步数
    scaler: 用于反标准化的scaler
    
    返回:
    predictions: 预测结果（原始尺度）
    """
    predictions = []
    last_sequence = scaled_data[-time_steps:]
    current_sequence = last_sequence.copy()

    for _ in range(future_steps):
        # 预测下一个值
        next_pred = model.predict(current_sequence.reshape(1, time_steps, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # 更新序列：移除第一个元素，添加预测值
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # 反标准化
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()

def main(data:pd.DataFrame, target_column:str, index_column:str, time_steps:int):
    '''
    单变量预测版本
    '''
    n_feature = 1  # 只有一个特征要预测

    # 可视化原始数据    
    plt.figure(figsize=(12, 6))
    plt.plot(data[index_column], data[target_column], label='Original Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Original Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    value = data[target_column].values

    print(f"数据统计信息:")
    print(f"均值: {np.mean(np.array(value)):.4f}")
    print(f"标准差: {np.std(np.array(value)):.4f}")
    print(f"最小值: {np.min(np.array(value)):.4f}")
    print(f"最大值: {np.max(np.array(value)):.4f}")

    # 数据预处理

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[target_column].values.reshape(-1, 1)).flatten()
    # print(scaled_data)
    print("数据预处理: ")
    print(f"原始数据范围:[{np.min(np.array(value)):.4f}, {np.max(np.array(value)):.4f}]")
    print(f"标准化后范围: [{np.min(scaled_data):.4f}, {np.max(scaled_data):.4f}]")

    # 创建训练和测试数据
    X, y = create_sequences(scaled_data, time_steps=time_steps)
    print(f"序列数据形状: X={X.shape}, y={y.shape}")

    train_rate = 0.8
    train_size = int(len(X) * train_rate)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    print(f"训练集大小: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集大小: X_test={X_test.shape}, y_test={y_test.shape}")

    # LSTM 的输入为 (batches, time_steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_feature))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_feature))
    
    print(f"重新调整后的形状:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")

    # 创建模型
    input_shape = (time_steps, n_feature)
    model = build_lstm_model(input_shape, n_feature=n_feature, units=[50, 50], dropout_rate=0.2)

    print("LSTM 模型结构：")
    model.summary()
    print("\n开始训练 LSTM 模型")

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1,
        shuffle=False  # 时间序列数据不建议打乱顺序
    )
    print("模型训练完成！")

    plot_history(history=history)

    # 模型预测
    print("进行模型预测...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # 反标准化预测结果
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 计算评估指标
    train_mse = mean_squared_error(y_train_actual, train_predictions)
    test_mse = mean_squared_error(y_test_actual, test_predictions)
    train_mae = mean_absolute_error(y_train_actual, train_predictions)
    test_mae = mean_absolute_error(y_test_actual, test_predictions)

    print(f"\n模型评估结果:")
    print(f"训练集 - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
    print(f"测试集 - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
    print(f"训练集 - RMSE: {np.sqrt(train_mse):.4f}")
    print(f"测试集 - RMSE: {np.sqrt(test_mse):.4f}")

    # 计算 R2 得分
    train_r2 = r2_score(y_train_actual, train_predictions)
    test_r2 = r2_score(y_test_actual, test_predictions)
    print(f"训练集 - R2: {train_r2:.4f}")
    print(f"测试集 - R2: {test_r2:.4f}")

    # 进行未来预测
    future_steps = 20
    print(f"使用最后{time_steps}个数据点预测未来{future_steps}个时间步...")
    future_predictions = predict_future(model, scaled_data, future_steps=future_steps, scaler=scaler, time_steps=time_steps)
    print(f"未来预测结果: {future_predictions}")

    # 创建未来日期
    last_date = data[index_column].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), 
                            periods=future_steps, freq='D')
    for i, (date, pred) in enumerate(zip(future_dates, future_predictions)):
        print(f"第{i+1}天 ({date.strftime('%Y-%m-%d')}): {pred:.4f}")

    plt.figure(figsize=(16, 8))

    # 绘制历史数据
    plt.plot(data[index_column], data[target_column], label='Historical Data', 
            linewidth=2, color='blue', marker='o', markersize=3)

    # 绘制未来预测
    plt.plot(future_dates, future_predictions, 
            label=f'Future Predictions ({future_steps} steps)', 
            linewidth=3, color='red', marker='s', markersize=4)

    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7, 
            label='Prediction Start')

    plt.xlabel('Date')
    plt.ylabel('TO BE MODIFIED!!!')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    step_for_avg = 10
    recent_avg = np.mean(np.array(data[target_column].values[-step_for_avg:]))
    future_avg = np.mean(future_predictions)
    trend_change = ((future_avg - recent_avg) / recent_avg) * 100

    print(f"\n趋势分析:")
    print(f"最近{step_for_avg}天平均值: {recent_avg:.4f}")
    print(f"未来预测平均值: {future_avg:.4f}")
    print(f"趋势变化: {trend_change:+.2f}%")

    if trend_change > 5:
        print("预测显示上升趋势")
    elif trend_change < -5:
        print("预测显示下降趋势")
    else:
        print("预测显示相对稳定的趋势")

    print(f"\nLSTM模型预测完成！")

if __name__ == '__main__':
    # 设置随机种子以确保结果可重现
    # log_stream = open("temp.txt", 'w', encoding='utf-8')
    # import sys
    # sys.stdout = log_stream
    np.random.seed(42)
    tf.random.set_seed(42)

    # 创建模拟时间序列数据
    years = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    trend = np.linspace(0, 100, 1000)
    seasonality = 5 * np.sin(np.linspace(0, 100*np.pi, 1000))
    noise = np.random.randn(1000) * 1.5
    tax_revenue = trend + seasonality + noise

    data = pd.DataFrame({
        'Date': years,
        'Tax_Revenue': tax_revenue
    })

    print("Original Tax Revenue Data:")
    print(data.head(10))
    print(f"\nData series length: {len(tax_revenue)}")

    main(data=data, target_column='Tax_Revenue', index_column='Date', time_steps=100)
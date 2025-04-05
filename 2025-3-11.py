import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

path = r"E:\常州普利司通_场站.csv"
df = pd.read_csv(path, usecols=["时间", " 辐照强度(Wh/㎡)", " 全场功率(kW)", " 环境温度(℃)"])
df["时间"] = pd.to_datetime(df["时间"])

# 处理辐照强度和功率：16:00到次日9:00缺失值填0
for column in [" 辐照强度(Wh/㎡)", " 全场功率(kW)"]:
    df.loc[(df["时间"].dt.hour >= 16) | (df["时间"].dt.hour < 9), column] = 0

    # 填充连续缺失值（大于等于4个的部分填0）
    missing = df[column].isnull()
    start = None
    for i in range(len(missing)):
        if missing[i] and start is None:
            start = i
        if not missing[i] and start is not None:
            if i - start >= 4:
                df.loc[start:i - 1, column] = 0
            start = None
    if start is not None and len(missing) - start >= 4:
        df.loc[start:, column] = 0

    # 取上下两个数的平均值填充剩余缺失值
    df[column] = df[column].fillna(df[column].rolling(2, min_periods=1).mean())
    df[column] = df[column].fillna(df[column].rolling(2, min_periods=1).mean())
    df[column] = df[column].fillna(df[column].rolling(2, min_periods=1).mean())
# 处理环境温度：每天用一元二次函数拟合填充缺失值
df["日期"] = df["时间"].dt.date
filled_df = pd.DataFrame()


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


for date, group in df.groupby("日期"):
    group = group.sort_values("时间").reset_index(drop=True)
    x = np.arange(len(group))
    y = group[" 环境温度(℃)"].values
    mask = group[" 环境温度(℃)"].isnull()

    if mask.any():
        try:
            params, _ = curve_fit(quadratic, x[~mask], y[~mask])
            y_pred = quadratic(x[mask], *params)
            group.loc[mask, " 环境温度(℃)"] = y_pred
        except:
            pass

    filled_df = pd.concat([filled_df, group], ignore_index=True)

# 移除临时日期列
filled_df.drop(columns=["日期"], inplace=True)

# 保存最终处理后的数据
output_path = r"E:\处理后数据.csv"
filled_df.to_csv(output_path, index=False, encoding="utf-8-sig")

# 计算最终剩余的缺失值个数
remaining_missing = filled_df.isnull().sum()
print("最终剩余的缺失值个数:")
print(remaining_missing)
print(f"处理后的数据已保存为: {output_path}")

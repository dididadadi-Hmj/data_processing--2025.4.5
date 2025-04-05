import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

'''
plt作图发现辐照和全场功率都和时间有关，夜晚是0，所以直接16-9点填成0应对大面积空缺的情况
填充后输出发现缺失值集中在十点和十五点，应该是不同月份日出问题，增加连续四个及以上的缺失值填充为0
之后再次输出发现缺失值大多不再连续，因数据过多，而且时间序列有一定的连续性，所以使用上下两个数据的平均值填充
温度缺失相对较少且分散，plt作图发现温度每天都类似于一元二次函数，以每天为单位，使用一元二次函数拟合填充
平均值填充部分复制了一下处理连续两个或三个空缺的问题
'''

path = r"E:\常州普利司通_场站.csv"  # 改为所使用的文件路径
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

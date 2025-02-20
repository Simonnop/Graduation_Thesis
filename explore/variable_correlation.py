import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os  # 添加os模块用于处理目录

# 设置Mac系统中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取当前使用的数据集名称
# dataset_name = 'Huron'  # 根据当前读取的数据文件设置
# dataset_name = 'Baikal'
dataset_name = 'Victoria'

# 读取数据
# df = pd.read_csv('./data/processed/Baikal_5min_merged.csv', low_memory=False)
df = pd.read_csv(f'./data/processed/{dataset_name}_5min_merged.csv', low_memory=False)
# df = pd.read_csv('./data/processed/Victoria_5min_merged.csv', low_memory=False)

# 合并数据集
# df = pd.concat([df_baikal, df_huron, df_victoria], ignore_index=True)

# 只选择数值类型的列进行相关性分析
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_columns]

# 计算相关性并按绝对值排序
corr = df_numeric.corr()['FanPower']
corr_abs = corr.abs().sort_values(ascending=False)
corr = corr[corr_abs.index]  # 保持原始相关系数，但按绝对值排序

# 准备特征翻译说明
feature_translations = {
    'Load': '整体建筑用电负荷',
    'AH1 SF CFM': 'AH1送风量',
    'AH1 RF CFM': 'AH1回风量',
    'AH1 RM5 CFM': 'AH1房间5气流量',
    'AH1 RM8 DUCT SP': 'AH1房间8风管静压',
    'AH1 DA TEMP': 'AH1送风温度',
    'AH1 RA TEMP': 'AH1回风温度',
    'AH1 MA TEMP': 'AH1混合空气温度',
    'AH1 HTG TEMP': 'AH1加热温度',
    'AH1 RM1 TEMP': 'AH1房间1温度',
    'AH1 RM5 STPT': 'AH1房间5设定温度',
    'AH1 RM5 VLV2 POS': 'AH1房间5阀门2位置',
    'AH1 RM6 VLV2 POS': 'AH1房间6阀门2位置',
    'AH1 RA HUM': 'AH1回风湿度',
    'AH2 SF CFM': 'AH2送风量',
    'AH2 RF CFM': 'AH2回风量',
    'AH2 RM4 CFM': 'AH2房间4气流量',
    'AH2 RM3 DUCT SP': 'AH2房间3风管静压',
    'AH2 MA TEMP': 'AH2混合空气温度',
    'AH2 HTG TEMP': 'AH2加热温度',
    'AH2 RM4 STPT': 'AH2房间4设定温度',
    'AH2 RM4 VLV2 POS': 'AH2房间4阀门2位置',
    'AH2 RM7 VLV2 POS': 'AH2房间7阀门2位置',
    'CLG LOOP3 LOAD TONS': '冷却回路3负荷吨数',
    'CLG LOOP5 CHWS FLOW': '冷却回路5冷冻水供水流量',
    'CLG LOOP5 CHWS TEMP': '冷却回路5冷冻水供水温度',
    'CLG LOOP5 CHWR TEMP': '冷却回路5冷冻水回水温度',
    'HTG LOOP3 LOAD KBTUS': '加热回路3负荷千BTU',
    'Event_Event ID': '事件ID',
    'Event_Down Change': '事件降低变化',
    'Event_Up Change': '事件提高变化'
}

# 创建保存结果的目录
os.makedirs(f'./figs/{dataset_name}', exist_ok=True)
os.makedirs('./docs', exist_ok=True)

# 显示并保存相关性结果
with open(f'./docs/correlation_results_{dataset_name}.txt', 'w', encoding='utf-8') as f:
    f.write(f"{dataset_name}数据集与风机功率(FanPower)最相关的30个特征（按相关性绝对值排序）:\n\n")
    for rank, (feature, value) in enumerate(corr[:30].items(), 1):
        translation = feature_translations.get(feature, feature)
        line = f"第{rank}名: {translation}\n({feature}): {value:.6f}\n"
        f.write(line)
        print(line)

# 绘制相关性热力图
plt.figure(figsize=(20, 15))
sns.heatmap(df_numeric[corr_abs[:30].index].corr(), annot=True, cmap='RdBu_r', center=0)
plt.title(f'{dataset_name}数据集前30个特征相关性热力图', fontsize=12, fontproperties='Arial Unicode MS')
plt.xticks(rotation=45, ha='right', fontproperties='Arial Unicode MS')
plt.yticks(fontproperties='Arial Unicode MS')
plt.tight_layout()
plt.savefig(f'./figs/{dataset_name}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制散点图
plt.figure(figsize=(15, 10))
for i, feature in enumerate(corr_abs[1:7].index):
    plt.subplot(2, 3, i+1)
    plt.scatter(df_numeric[feature], df_numeric['FanPower'], alpha=0.5)
    translation = feature_translations.get(feature, feature)
    plt.xlabel(f"{translation}\n({feature})", fontsize=10, fontproperties='Arial Unicode MS')
    plt.ylabel('风机功率 (FanPower)', fontsize=10, fontproperties='Arial Unicode MS')
    plt.title(f'相关性: {corr[feature]:.3f}', fontsize=11, fontproperties='Arial Unicode MS')
plt.tight_layout()
plt.savefig(f'./figs/{dataset_name}/correlation_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

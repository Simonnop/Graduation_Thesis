import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os  # 添加os模块用于处理目录

import platform
import matplotlib.font_manager as fm

# 设置中文字体
def set_chinese_font():
    system = platform.system()
    
    # 获取系统上所有可用的字体
    font_names = [f.name for f in fm.fontManager.ttflist]
    
    # 根据不同操作系统设置默认字体
    if system == 'Darwin':  # macOS
        if 'Arial Unicode MS' in font_names:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        elif 'PingFang SC' in font_names:
            plt.rcParams['font.sans-serif'] = ['PingFang SC']
        elif 'Heiti SC' in font_names:
            plt.rcParams['font.sans-serif'] = ['Heiti SC']
    elif system == 'Windows':  # Windows
        if 'Microsoft YaHei' in font_names:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        elif 'SimHei' in font_names:
            plt.rcParams['font.sans-serif'] = ['SimHei']
    elif system == 'Linux':  # Linux
        if 'WenQuanYi Micro Hei' in font_names:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        elif 'Noto Sans CJK JP' in font_names:
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
        elif 'Noto Sans CJK SC' in font_names:
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
        elif 'DejaVu Sans' in font_names:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 设置负号显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 增大字体大小（进一步增大）
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16

# 调用字体设置函数
set_chinese_font()
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

# 添加特征聚合功能
def aggregate_similar_features(df):
    """对相似特征进行聚合，如不同AH的相同类型数据"""
    # 创建一个新的DataFrame来存储聚合后的特征
    df_agg = pd.DataFrame(index=df.index)
    
    # 预处理列名，移除VICTORIA前缀
    df_columns = df.columns.tolist()
    rename_dict = {}
    for col in df_columns:
        if col.startswith('VICTORIA '):
            rename_dict[col] = col.replace('VICTORIA ', '')
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # 定义需要聚合的特征组
    feature_groups = {
        'MA_TEMP_SUM': ['AH1 MA TEMP', 'AH2 MA TEMP', 'AH3 MA TEMP'],  # 混合空气温度总和
        'DA_TEMP_SUM': ['AH1 DA TEMP', 'AH2 DA TEMP', 'FCU1 DA TEMP'],  # 送风温度总和
        'RA_TEMP_SUM': ['AH1 RA TEMP', 'AH2 RA TEMP', 'AH3 RA TEMP'],  # 回风温度总和
        'RM_STPT_SUM': ['AH1 RM5 STPT', 'AH2 RM4 STPT', 'RM3 STPT'],  # 房间设定温度总和
        'RM_TEMP_SUM': ['RM3 TEMP', 'RM4 TEMP'],  # 房间温度总和
    }
    
    # 计算每组特征的总和
    for new_feature, features in feature_groups.items():
        # 检查所有特征是否都存在于数据集中
        valid_features = [f for f in features if f in df.columns]
        if valid_features:
            df_agg[new_feature] = df[valid_features].sum(axis=1)
    
    # 添加非AH和RM的原始特征，但排除Load、冷却系统、排风扇和特定事件ID
    excluded_patterns = ['AH1', 'AH2', 'AH3', 'RM', 'FCU', 'Load', 'CLG', 'CHW', 'EXF', 'Event_Event ID']
    non_ah_rm_features = [col for col in df.columns if not any(x in col for x in excluded_patterns)]
    for feature in non_ah_rm_features:
        df_agg[feature] = df[feature]
    
    # 确保FanPower被保留
    if 'FanPower' in df.columns:
        df_agg['FanPower'] = df['FanPower']
    
    return df_agg

# 应用特征聚合
df = aggregate_similar_features(df)

# 只选择数值类型的列进行相关性分析
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_columns]

# 计算相关性并按绝对值排序
corr = df_numeric.corr()['FanPower']
corr_abs = corr.abs().sort_values(ascending=False)
corr = corr[corr_abs.index]  # 保持原始相关系数，但按绝对值排序

# 准备特征翻译说明
feature_translations = {
    'FanPower': '风机功率',
    'SF_CFM_SUM': '送风量总和',
    'RF_CFM_SUM': '回风量总和',
    'RM_CFM_SUM': '房间气流量总和',
    'MA_TEMP_SUM': '混合空气温度总和',
    'DA_TEMP_SUM': '送风温度总和',
    'RA_TEMP_SUM': '回风温度总和',
    'RM_STPT_SUM': '房间设定温度总和',
    'RM_TEMP_SUM': '房间温度总和',
    'CLG CHW FLOW': '冷却冷冻水流量',
    'EXF2 CFM': '排风扇2气流量',
    'CLG PLANTDIST CHWR TEMP': '冷却系统冷冻水回水温度',
    'CLG CHWR TEMP': '冷却冷冻水回水温度',
    'OA TEMP1': '室外空气温度',
    'CLG PLANTDIST CHW FLOW': '冷却系统冷冻水流量',
    'Event_Down Change': '上升变化',
    'Event_Event ID': '需求响应事件',
    'Event_Up Change': '下降变化',
}

# 更新特征翻译
feature_translations.update({
    'MA_TEMP_SUM': '混合空气温度',
    'DA_TEMP_SUM': '送风温度',
    'RA_TEMP_SUM': '回风温度',
    'RM_STPT_SUM': '温度设定点',
    'RM_TEMP_SUM': '室内温度',
    'OA WB TEMP': '室外湿球温度',
})

# 创建保存结果的目录
os.makedirs(f'./figs/{dataset_name}', exist_ok=True)
os.makedirs('./docs', exist_ok=True)

# 显示并保存相关性结果
with open(f'./docs/correlation_results_{dataset_name}.txt', 'w', encoding='utf-8') as f:
    f.write(f"{dataset_name}数据集与风机功率(FanPower)最相关的20个特征（按相关性绝对值排序）:\n\n")
    for rank, (feature, value) in enumerate(corr[:20].items(), 1):
        translation = feature_translations.get(feature, feature)
        line = f"第{rank}名: {translation}\n({feature}): {value:.6f}\n"
        f.write(line)
        print(line)

# 绘制相关性热力图
plt.figure(figsize=(15, 12))
# 获取前10个特征的中文名称
top10_features = corr_abs[:10].index
top10_translations = [feature_translations.get(feature, feature) for feature in top10_features]

# 计算相关性矩阵并取绝对值
corr_matrix = df_numeric[top10_features].corr().abs()

# 使用中文标签绘制热力图
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            xticklabels=top10_translations, yticklabels=top10_translations)
plt.title(f' ')
plt.xticks(rotation=45, ha='right')
plt.yticks()
plt.tight_layout()
plt.savefig(f'./figs/{dataset_name}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制散点图 - 修改为包含新的聚合特征
plt.figure(figsize=(15, 10))
# 获取前6个相关性最高的特征（包括原始和聚合特征）
top_features = corr_abs[1:7].index
for i, feature in enumerate(top_features):
    plt.subplot(2, 3, i+1)
    plt.scatter(df_numeric[feature], df_numeric['FanPower'], alpha=0.5)
    translation = feature_translations.get(feature, feature)
    plt.xlabel(f"{translation}\n({feature})")
    plt.ylabel('风机功率 (FanPower)')
    plt.title(f'相关性: {corr[feature]:.3f}')
plt.tight_layout()
plt.savefig(f'./figs/{dataset_name}/correlation_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 额外添加一个只展示聚合特征与FanPower相关性的图表
agg_features = [f for f in corr.index if f.endswith('_SUM')]
if agg_features:
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(agg_features[:min(6, len(agg_features))]):
        plt.subplot(2, 3, i+1)
        plt.scatter(df_numeric[feature], df_numeric['FanPower'], alpha=0.5)
        translation = feature_translations.get(feature, feature)
        plt.xlabel(f"{translation}\n({feature})")
        plt.ylabel('风机功率 (FanPower)')
        plt.title(f'相关性: {corr[feature]:.3f}')
    plt.tight_layout()
    plt.savefig(f'./figs/{dataset_name}/agg_features_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

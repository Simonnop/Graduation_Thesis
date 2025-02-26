import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import platform
import matplotlib.font_manager as fm

# 全局变量声明
all_times = None
data_name = None

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

# 调用字体设置函数
set_chinese_font()

def plot_forecast(target_time = '2021-09-17 09:00:00', model_name='LSTM', dataset=None):
    # 将输入的时间字符串转换为datetime对象
    target_time = pd.to_datetime(target_time)
    target_date = target_time.strftime('%Y-%m-%d')
    
    # 获取目标时间前12小时到后4小时的数据
    start_time = target_time - pd.Timedelta(hours=18)
    end_time = target_time + pd.Timedelta(hours=4)
    mask = (all_times.index >= start_time) & (all_times.index <= end_time)
    plot_data = all_times[mask]

    # 创建主图
    fig = plt.figure(figsize=(15, 8))
    main_ax = plt.gca()
    main_ax.set_facecolor('#EAEAF1')  # 浅蓝色背景
    
    # 添加网格，设置为白色
    main_ax.grid(True, color='white', linewidth=3, alpha=0.7, zorder=0)
    
    # 为主图添加边框
    for spine in main_ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    
    # 绘制主图的实际数据
    main_ax.plot(plot_data.index, plot_data['FanPower'], 'b-', alpha=0.7, label='实际功率', linewidth=4)
    
    # 添加目标时间点的垂直线
    main_ax.axvline(x=target_time, color='black', linestyle='--', alpha=0.5)
    main_ax.text(target_time, main_ax.get_ylim()[1], '预测时点', 
                rotation=90, verticalalignment='top', color='black', fontsize=18)

    try:
        # 从results目录读取预测结果
        pred_path = f'results/{data_name}/{model_name}/predictions.npy'
        trues_path = f'results/{data_name}/{model_name}/true_values.npy'
        
        if os.path.exists(pred_path) and os.path.exists(trues_path):
            # 加载预测结果和真实值
            all_preds = np.load(pred_path)
            all_trues = np.load(trues_path)
            
            # 获取目标时间后4小时的实际数据（确保只获取48个点）
            target_data = all_times.loc[target_time:target_time + pd.Timedelta(hours=4)]['FanPower'].values[:48]
            
            # 重新整理真实值序列，每48个点为一组
            num_sequences = len(all_trues) // 48
            all_trues = all_trues.reshape(num_sequences, 48)
            all_preds = all_preds.reshape(num_sequences, 48)
            
            # 在all_trues中寻找匹配的序列
            for i in range(len(all_trues)):
                if np.allclose(all_trues[i], target_data, equal_nan=True):
                    pred = all_preds[i]
                    
                    # 生成预测时间点
                    pred_times = pd.date_range(start=target_time, periods=len(pred), freq='5min')
                    
                    # 在主图中绘制预测结果
                    main_ax.plot(pred_times, pred.reshape(-1), 'r-', alpha=0.7, label='预测功率', linewidth=4)
                    
                    # 获取这段时间的真实值用于对比
                    future_mask = (all_times.index > target_time) & (all_times.index <= end_time)
                    future_actual = all_times[future_mask]
                    
                    # 创建子图
                    inset_position = [0.17, 0.45, 0.35, 0.35]  # [left, bottom, width, height]
                    inset_ax = fig.add_axes(inset_position)
                    
                    # 为子图添加边框
                    for spine in inset_ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.5)
                        spine.set_visible(True)
                    
                    # 绘制放大视图中的数据
                    inset_ax.plot(pred_times, pred.reshape(-1), 'r-', alpha=0.7, label='预测功率', linewidth = 3)
                    inset_ax.plot(future_actual.index, future_actual['FanPower'], 'b-', alpha=0.7, label='未来实际功率', linewidth = 3)
                    
                    # 设置放大子图的格式
                    inset_ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))  # 每30分钟显示一个时间标签
                    inset_ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))   # 每30分钟显示一个小刻度
                    inset_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    inset_ax.grid(True, which='major', alpha=0.3)  # 主网格线
                    inset_ax.grid(True, which='minor', alpha=0.15)  # 次网格线
                    inset_ax.set_title('预测时段', fontsize=10, pad=10)
                    inset_ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

                    # 获取预测时段的y轴范围
                    if not future_actual.empty:
                        main_y_min = min(future_actual['FanPower'].min(), pred.min())
                        main_y_max = max(future_actual['FanPower'].max(), pred.max())
                        # 添加一些边距
                        y_margin = (main_y_max - main_y_min) * 0.1
                        main_y_min -= y_margin
                        main_y_max += y_margin
                        
                        # 设置子图的y轴范围
                        inset_ax.set_ylim(main_y_min, main_y_max)
                        
                        # 添加连接线
                        from matplotlib.patches import ConnectionPatch, Rectangle

                        # 获取主图中预测时段的数据范围
                        zoom_start = target_time
                        zoom_end = target_time + pd.Timedelta(hours=4)
                        
                        # 获取预测时段的数据范围
                        pred_mask = (plot_data.index >= zoom_start) & (plot_data.index <= zoom_end)
                        pred_data = plot_data[pred_mask]
                        
                        # 获取预测时段的y轴范围
                        if not pred_data.empty:
                            main_y_min = min(pred_data['FanPower'].min(), pred.min())
                            main_y_max = max(pred_data['FanPower'].max(), pred.max())
                            # 添加一些边距
                            y_margin = (main_y_max - main_y_min) * 0.1
                            main_y_min -= y_margin
                            main_y_max += y_margin
                            
                            # 在主图中添加矩形框标记预测时段
                            rect = Rectangle(
                                (mdates.date2num(zoom_start), main_y_min),
                                mdates.date2num(zoom_end) - mdates.date2num(zoom_start),
                                main_y_max - main_y_min,
                                fill=False,
                                edgecolor='black',
                                linestyle='-',
                                linewidth=1,
                                zorder=5
                            )
                            main_ax.add_patch(rect)
                            
                            # 设置子图的y轴范围与主图中框选区域一致
                            inset_ax.set_ylim(main_y_min, main_y_max)
                            
                            # 创建两条对角连接线
                            connections = [
                                # 左上角到右上角
                                ((mdates.date2num(zoom_start), main_y_max), 
                                 (inset_ax.get_xlim()[1], inset_ax.get_ylim()[1])),
                                # 左下角到右下角
                                ((mdates.date2num(zoom_start), main_y_min), 
                                 (inset_ax.get_xlim()[1], inset_ax.get_ylim()[0]))
                            ]
                            
                            # 添加连接线
                            for corner_main, corner_inset in connections:
                                con = ConnectionPatch(
                                    xyA=corner_main,
                                    xyB=corner_inset,
                                    coordsA="data",
                                    coordsB="data",
                                    axesA=main_ax,
                                    axesB=inset_ax,
                                    color="black",
                                    linestyle="-",
                                    linewidth=1,
                                    zorder=5
                                )
                                main_ax.add_artist(con)

                    # 标记需求响应事件（在子图中）
                    event_periods = plot_data[plot_data['event_type'] != 'N']
                    event_colors = {'UD': 'red', 'DU': 'blue', 'U': 'green'}
                    
                    if not event_periods.empty:
                        for event_id in event_periods['event_id'].unique():
                            event_data = event_periods[event_periods['event_id'] == event_id]
                            event_start = event_data.index[0]
                            event_end = event_data.index[-1]
                            
                            # 检查事件是否与预测时段有重叠
                            if not (event_end < zoom_start or event_start > zoom_end):
                                event_type = event_data['event_type'].iloc[0]
                                color = event_colors.get(event_type, 'gray')
                                
                                # 计算事件在预测时段内的起止时间
                                plot_start = max(event_start, zoom_start)
                                plot_end = min(event_end, zoom_end)
                                
                                inset_ax.axvspan(plot_start, plot_end, 
                                               alpha=0.2, color=color)
        
    except Exception as e:
        import traceback
        print(f"预测出错: {str(e)}")
        print("详细错误信息:")
        print(traceback.format_exc())

    # 标记需求响应事件（在主图中）
    event_periods = plot_data[plot_data['event_type'] != 'N']
    event_colors = {'UD': 'red', 'DU': 'blue', 'U': 'green'}
    added_event_types = set()

    if not event_periods.empty:
        for event_id in event_periods['event_id'].unique():
            event_data = event_periods[event_periods['event_id'] == event_id]
            event_type = event_data['event_type'].iloc[0]
            color = event_colors.get(event_type, 'gray')
            
            if event_type not in added_event_types:
                main_ax.axvspan(event_data.index[0], event_data.index[-1], 
                             alpha=0.2, color=color, 
                             label=f'{event_type}类型事件')
                added_event_types.add(event_type)
            else:
                main_ax.axvspan(event_data.index[0], event_data.index[-1], 
                             alpha=0.2, color=color)

    # 设置主图的标题和标签
    main_ax.set_title(f'建筑总功率 ({target_time.strftime("%Y-%m-%d %H:%M")})')
    main_ax.set_xlabel('时间')
    main_ax.set_ylabel('功率 (kW)')
    
    # 调整主图的图例位置和显示
    main_ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')

    # 设置主图x轴格式
    main_ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    main_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

def load_data(file_path):
    """加载并预处理数据"""
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 将时间列转换为datetime格式
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 确保数据按时间排序
    df.sort_index(inplace=True)
    
    return df

def process_dr_events(df):
    """处理需求响应事件数据"""
    # 事件类型列名映射
    event_type_mapping = {
        'Event_Type': 'event_type',
        'EventType': 'event_type',
        'type': 'event_type',
        'Type': 'event_type'
    }
    
    # 事件ID列名映射
    event_id_mapping = {
        'Event_Event ID': 'event_id',
        'EventID': 'event_id',
        'id': 'event_id',
        'ID': 'event_id'
    }
    
    # 变化值列名映射
    change_mapping = {
        'Event_Up Change': 'up_change',
        'Event_Down Change': 'down_change',
        'UpChange': 'up_change',
        'DownChange': 'down_change'
    }
    
    # 尝试重命名列
    for mappings in [event_type_mapping, event_id_mapping, change_mapping]:
        for old_name, new_name in mappings.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename(columns={old_name: new_name})
    
    # 初始化缺失的列
    if 'event_type' not in df.columns:
        df['event_type'] = 'N'  # 默认为无事件
    if 'event_id' not in df.columns:
        df['event_id'] = ''
    if 'up_change' not in df.columns:
        df['up_change'] = 0.0
    if 'down_change' not in df.columns:
        df['down_change'] = 0.0
    
    return df

def main(data_name, target_time, model_name='LSTM'):
    """主函数"""
    
    try:
        # 设置全局变量
        global all_times, data

        data = data_name
        
        # 构建数据文件路径
        data_path = f'data/processed/{data_name}_5min_merged.csv'
        
        # 创建图片保存目录
        target_date = pd.to_datetime(target_time).strftime('%Y-%m-%d')
        save_dir = f'figs/{data_name}/predict/{target_date}'
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载数据
        df = load_data(data_path)
        
        # 处理需求响应事件数据
        df = process_dr_events(df)
        
        # 设置全局变量的值
        all_times = df
        
        # 获取所有有需求响应事件的日期
        event_dates = all_times[all_times['event_type'] != 'N'].index.strftime('%Y-%m-%d').unique()
        
        if target_date not in event_dates:
            print("不是需求响应事件")
            print(f"找到以下日期的需求响应事件：")
            for i, date in enumerate(event_dates, 1):
                print(f"{i}. {date}")
            return
            
        # 调用绘图函数
        plot_forecast(target_time, model_name)
        
        # 保存图片
        save_path = os.path.join(save_dir, f'{model_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == '__main__':
    target_time_list = ['2021-08-13 09:00:00',
                        '2021-09-06 12:00:00',
                        '2021-09-08 12:00:00',
                        '2021-09-09 12:00:00',
                        '2021-09-17 12:00:00',
                        '2021-09-20 12:00:00',
                        '2021-09-22 12:00:00',
                        '2021-09-27 12:00:00']
    data_name = 'Huron'
    model_list = ['DLinearGRU', 'GRU', 'Transformer', 'DLinear','DLinearformer']
    
    for target_time in target_time_list:
        for model_name in model_list:
            print(f"正在处理 {target_time} 的 {model_name} 模型预测结果...")
            main(data_name, target_time, model_name)

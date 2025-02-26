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

# 设置Mac系统中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_forecast(target_time = '2021-09-17 09:00:00', model=None, dataset=None):
    # 将输入的时间字符串转换为datetime对象
    target_time = pd.to_datetime(target_time)
    
    # 获取目标时间前12小时到后4小时的数据
    start_time = target_time - pd.Timedelta(hours=18)
    end_time = target_time + pd.Timedelta(hours=4)
    mask = (all_times.index >= start_time) & (all_times.index <= end_time)
    plot_data = all_times[mask]

    # 创建主图
    fig = plt.figure(figsize=(15, 8))
    main_ax = plt.gca()
    main_ax.set_facecolor('#EAEAF1')  # 浅蓝色背景
    # fig.patch.set_facecolor('#EAEAF1')  # 图形整体背景也设置为浅蓝色
    # 添加网格，设置为白色
    main_ax.grid(True, color='white', linewidth=3, alpha=0.7, zorder=0)
    
    
    # 为主图添加边框
    for spine in main_ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    
    # 绘制主图的实际数据
    main_ax.plot(plot_data.index, plot_data['power'], 'b-', alpha=0.7, label='实际功率', linewidth=4)
    
    # 添加目标时间点的垂直线
    main_ax.axvline(x=target_time, color='black', linestyle='--', alpha=0.5)
    main_ax.text(target_time, main_ax.get_ylim()[1], '预测时点', 
                rotation=90, verticalalignment='top', color='black', fontsize=18)

    # 如果提供了模型和数据集，添加预测结果
    if model is not None and dataset is not None:
        try:
            # 获取输入数据（目标时间点前的lookback个数据点）
            input_data = all_times.loc[:target_time].tail(dataset.lookback)
            print(f"输入数据长度: {len(input_data)}, 需要长度: {dataset.lookback}")
            
            if len(input_data) >= dataset.lookback:
                # 构建输入特征
                x_power = dataset.power_scaler.transform(input_data['power'].values.reshape(-1, 1))
                x_hour = np.array(input_data.index.hour, dtype=float) / 24.0
                x_day = np.array(input_data.index.dayofweek, dtype=float) / 7.0
                x_event = dataset.event_encoder.transform(input_data['event_type']) / len(dataset.event_encoder.classes_)
                x_changes = dataset.change_scaler.transform(np.column_stack((input_data['up_change'], input_data['down_change'])))
                
                print(f"特征形状: power={x_power.shape}, hour={x_hour.shape}, day={x_day.shape}, "
                      f"event={x_event.shape}, changes={x_changes.shape}")
                
                # 组合特征
                x = np.column_stack((
                    x_power,
                    x_hour.reshape(-1, 1),
                    x_day.reshape(-1, 1),
                    x_event.reshape(-1, 1),
                    x_changes
                ))
                
                print(f"组合后的输入特征形状: {x.shape}")
                
                # 进行预测
                model.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(x)
                    print(f"转换为tensor后的形状: {x.shape}")
                    
                    # 将输入数据移到与模型相同的设备上
                    device = next(model.parameters()).device
                    print(f"模型所在设备: {device}")
                    x = x.to(device)
                    
                    pred = model(x.unsqueeze(0))
                    print(f"预测结果形状: {pred.shape}")
                    
                    # 将预测结果移回CPU进行后续处理
                    pred = pred.cpu()
                    pred = dataset.inverse_transform_power(pred.numpy())
                    print(f"转换回实际值后的预测结果形状: {pred.shape}")
                
                # 生成预测时间点
                pred_times = pd.date_range(start=target_time, periods=len(pred)+1, freq='5min')[1:]
                print(f"预测时间点数量: {len(pred_times)}")
                
                # 在主图中绘制预测结果
                main_ax.plot(pred_times, pred.reshape(-1), 'r-', alpha=0.7, label='预测功率', linewidth=4)
                
                # 获取这段时间的真实值用于对比
                future_mask = (all_times.index > target_time) & (all_times.index <= end_time)
                future_actual = all_times[future_mask]
                # main_ax.plot(future_actual.index, future_actual['power'], 'b-', alpha=0.5, label='未来实际功率')

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
                inset_ax.plot(future_actual.index, future_actual['power'], 'b-', alpha=0.7, label='未来实际功率', linewidth = 3)
                
                # 设置放大子图的格式
                # 设置放大子图的格式
                # 设置放大子图的格式
                inset_ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))  # 每10分钟显示一个时间标签
                inset_ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))   # 每5分钟显示一个小刻度
                inset_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                inset_ax.grid(True, which='major', alpha=0.3)  # 主网格线
                inset_ax.grid(True, which='minor', alpha=0.15)  # 次网格线
                inset_ax.set_title('预测时段', fontsize=10, pad=10)
                inset_ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

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
                    main_y_min = min(pred_data['power'].min(), pred.min())
                    main_y_max = max(pred_data['power'].max(), pred.max())
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
    # main_ax.grid(True)
    
    # 调整主图的图例位置和显示
    main_ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')

    # 设置主图x轴格式
    main_ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    main_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 调整布局
    # plt.tight_layout()
    # plt.show()

def load_data(file_path):
    """加载并预处理数据"""
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 打印列名以便调试
    print("数据文件中的列名:", df.columns.tolist())
    
    # 将时间列转换为datetime格式
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 确保数据按时间排序
    df.sort_index(inplace=True)
    
    # 重命名功率相关的列
    power_column_mapping = {
        'Load': 'power',  # 添加 Load 到映射中
        'kW': 'power',
        'value': 'power',
        'Power': 'power',
        'power_kW': 'power'
    }
    
    # 尝试重命名功率列
    for old_name, new_name in power_column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: 'power'})
            print(f"将列 '{old_name}' 重命名为 'power'")
            break
    
    # 如果仍然没有找到功率列，抛出错误
    if 'power' not in df.columns:
        raise ValueError(f"未找到功率数据列。可用的列名有: {df.columns.tolist()}")
    
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

def main(data_name, target_time):
    """主函数"""
    
    try:
        # 构建数据文件路径
        data_path = f'data/processed/{data_name}_5min_merged.csv'
        
        # 创建图片保存目录
        save_dir = f'figs/{data_name}/predict'
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载数据
        print(f"正在加载数据文件: {data_path}")
        df = load_data(data_path)
        
        # 处理需求响应事件数据
        df = process_dr_events(df)
        
        # 设置全局变量供plot_forecast函数使用
        global all_times
        all_times = df
        
        # 获取所有有需求响应事件的日期
        event_dates = all_times[all_times['event_type'] != 'N'].index.strftime('%Y-%m-%d').unique()
        
        target_date = pd.to_datetime(target_time).strftime('%Y-%m-%d')
        if target_date not in event_dates:
            print("不是需求响应事件")
            print(f"找到以下日期的需求响应事件：")
            for i, date in enumerate(event_dates, 1):
                print(f"{i}. {date}")
            return
            
        print(f"\n正在显示 {target_time} 的需求响应事件可视化...")
        
        # 调用绘图函数
        plot_forecast(target_time)
        
        # 保存图片
        save_path = os.path.join(save_dir, f'{target_date}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == '__main__':
    target_time = '2021-07-27 09:00:00'
    data_name = 'Victoria'
    # data_name = 'Huron'
    main(data_name, target_time)

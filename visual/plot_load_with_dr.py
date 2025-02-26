import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
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


def load_and_process_data(data_path):
    """加载并处理数据"""
    # 读取CSV文件
    df = pd.read_csv(data_path, low_memory=False)
    
    # 将时间列转换为datetime索引
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 确保所需列存在
    # 重命名列
    column_mapping = {
        'FanPower': 'power',
        'Event_Event ID': 'event_id',
        'Event_Type': 'event_type', 
        'Event_Up Change': 'up_change',
        'Event_Down Change': 'down_change'
    }
    
    # 重命名列
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # 检查必需的列
    required_columns = ['power', 'event_type', 'event_id', 'up_change', 'down_change'] 
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据中缺少必需的列: {col}")
    
    return df

# 选择一个有需求响应事件的日期进行可视化
def plot_single_day_dr(target_time, data_name=None):  # 修改参数名以更清晰
    # 确保使用正确的日期格式
    target_date = pd.to_datetime(target_time).strftime('%Y-%m-%d')
    
    # 获取目标日期的数据
    daily_mask = all_times.index.strftime('%Y-%m-%d') == target_date
    daily_data = all_times[daily_mask]

    # 检查数据是否为空
    if daily_data.empty:
        print(f"警告：{target_date} 没有找到数据")
        return

    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))

    # 绘制全天数据，进一步增加线条粗细
    ax.plot(daily_data.index, daily_data['power'], 'b-', alpha=0.7, label='总功率', linewidth=3.5)

    # 标记需求响应事件区间，根据类型使用不同颜色
    event_periods = daily_data[daily_data['event_type'] != 'N']
    event_colors = {'UD': 'red', 'DU': 'blue', 'U': 'green'}
    
    # 记录已添加到图例的事件类型
    legend_added = set()

    # 标记事件
    for event_id in event_periods['event_id'].unique():
        event_data = event_periods[event_periods['event_id'] == event_id]
        event_type = event_data['event_type'].iloc[0]
        color = event_colors.get(event_type, 'gray')
        
        # 绘制事件区间，只有第一次出现时添加标签
        label = f'{event_type}类型事件' if event_type not in legend_added else None
        ax.axvspan(event_data.index[0], event_data.index[-1], 
                    alpha=0.2, color=color, 
                    label=label)
        
        # 记录已添加到图例的事件类型
        legend_added.add(event_type)
        
        # 添加事件信息标注，进一步增大字体大小
        event_info = event_data.iloc[0]
        info_text = (f'类型：{event_type}\n'
                    f'上调：{event_info["up_change"]:.1f}\n下调：{event_info["down_change"]:.1f}')
        
        ax.text(event_data.index[0], ax.get_ylim()[1] * 0.1, info_text,
                horizontalalignment='left',
                verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color),
                fontsize=16)

    # 设置图表标签（使用空标题）
    ax.set_title(" ", fontsize=20)
    ax.set_xlabel('时间', fontsize=18)
    ax.set_ylabel('功率 (kW)', fontsize=18)
    ax.grid(True)
    ax.legend(fontsize=16)

    # 设置x轴格式
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 调整布局
    plt.tight_layout()
    
    # 创建保存图片的文件夹（如果不存在）
    save_dir = f'figs/{data_name}/load'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图片，使用日期作为文件名
    save_path = f'{save_dir}/{target_date}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'图片已保存至: {save_path}')
    
    # plt.show()

def main(data_name, target_time):
    """主函数，组织数据加载和可视化流程"""
    try:
        # 构建数据文件路径
        data_path = f'data/processed/{data_name}_5min_merged.csv'
        
        # 创建图片保存目录
        save_dir = f'figs/{data_name}'
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载数据
        print(f"正在加载数据文件: {data_path}")
        global all_times  # 使plot_single_day_dr函数可以访问数据
        all_times = load_and_process_data(data_path)
        
        # 获取所有有需求响应事件的日期
        event_dates = all_times[all_times['event_type'] != 'N'].index.strftime('%Y-%m-%d').unique()
        
        # 添加调试信息
        print(f"数据时间范围: {all_times.index.min()} 到 {all_times.index.max()}")
        print(f"数据总行数: {len(all_times)}")
        
        target_date = pd.to_datetime(target_time).strftime('%Y-%m-%d')
        daily_data = all_times[all_times.index.strftime('%Y-%m-%d') == target_date]
        print(f"目标日期 {target_date} 的数据行数: {len(daily_data)}")
        
        if target_date not in event_dates:
            print("不是需求响应事件")
            print(f"找到以下日期的需求响应事件：")
            for i, date in enumerate(event_dates, 1):
                print(f"{i}. {date}")
            return
            
        print(f"\n正在显示 {target_time} 的需求响应事件可视化...")
        plot_single_day_dr(target_time, data_name)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    target_time = '2021-09-17'
    # data_name = 'Victoria'
    data_name = 'Huron'
    main(data_name, target_time)
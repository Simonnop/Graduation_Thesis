import pandas as pd
import numpy as np

def load_and_process_data(file_path='data/processed/Victoria_5min_merged.csv'):
    """
    加载并处理数据,根据表2-1的分类进行汇总
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 创建结果DataFrame
    result = pd.DataFrame()
    
    # 1. 基础信息
    base_cols = {
        'timestamp': 'timestamp',  # 时间戳
        'status': 'Status',       # 系统状态
        'fan_power': 'FanPower'   # 风机功率
    }
    
    # 2. 温度数据
    temp_cols = {
        'room_temp': [col for col in df.columns if 'RM' in col and 'TEMP' in col],      # 室内温度
        'ma_temp': [col for col in df.columns if 'MA TEMP' in col],                     # 混合空气温度
        'oa_temp': [col for col in df.columns if 'OA TEMP' in col and 'WB' not in col], # 室外空气温度
        'oa_wb_temp': [col for col in df.columns if 'OA WB TEMP' in col]                # 室外湿球温度
    }
    
    # 3. 空调数据
    hvac_cols = {
        'temp_setpoint': [col for col in df.columns if 'STPT' in col and 'TEMP' in col], # 温度设定点
        'flow_setpoint': [col for col in df.columns if 'STPT' in col and 'CFM' in col],  # 风量设定点
        'ra_temp': [col for col in df.columns if 'RA TEMP' in col],                      # 回风温度
        'ra_flow': [col for col in df.columns if 'RA CFM' in col],                       # 回风风量
        'sa_temp': [col for col in df.columns if 'SA TEMP' in col],                      # 送风温度
        'sa_flow': [col for col in df.columns if 'SA CFM' in col]                        # 送风风量
    }
    
    # 4. 需求响应数据
    dr_cols = {
        'event_id': 'Event_Event ID',      # 事件ID
        'event_type': 'Event_Type',        # 事件类型
        'up_change': 'Event_Up Change',    # 上升变化
        'down_change': 'Event_Down Change' # 下降变化
    }
    
    # 处理基础信息
    for new_col, old_col in base_cols.items():
        if old_col in df.columns:
            result[new_col] = df[old_col]
    
    # 处理温度和空调数据
    for cols_dict in [temp_cols, hvac_cols]:
        for new_col, cols in cols_dict.items():
            valid_cols = [col for col in cols if col in df.columns]
            if valid_cols:
                result[new_col] = df[valid_cols].mean(axis=1)
    
    # 处理需求响应数据
    for new_col, old_col in dr_cols.items():
        if old_col in df.columns:
            result[new_col] = df[old_col]
    
    # 添加时间特征
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    
    # 计算缺失值统计
    missing_stats = result.isnull().sum()
    
    return result, missing_stats

def print_data_summary(processed_data, missing_stats):
    """
    打印数据摘要信息
    """
    print("\n数据基本信息:")
    print(f"总记录数: {len(processed_data)}")
    print(f"时间范围: {processed_data['timestamp'].min()} 到 {processed_data['timestamp'].max()}")
    print(f"变量数量: {len(processed_data.columns)}")
    
    # 1. 基础信息统计
    print("\n基础信息统计:")
    base_cols = [col for col in ['status', 'fan_power'] if col in processed_data.columns]
    if base_cols:
        print(processed_data[base_cols].describe())
    
    # 2. 温度数据统计
    print("\n温度数据统计(°F):")
    temp_cols = [col for col in ['room_temp', 'ma_temp', 'oa_temp', 'oa_wb_temp'] 
                if col in processed_data.columns]
    if temp_cols:
        print(processed_data[temp_cols].describe())
    
    # 3. 空调数据统计
    print("\n空调数据统计:")
    hvac_cols = [col for col in ['temp_setpoint', 'flow_setpoint', 'ra_temp', 'ra_flow', 
                                'sa_temp', 'sa_flow'] 
                if col in processed_data.columns]
    if hvac_cols:
        print(processed_data[hvac_cols].describe())
    
    # 4. 需求响应数据统计
    print("\n需求响应事件统计:")
    if 'event_type' in processed_data.columns:
        print(processed_data['event_type'].value_counts())
    
    print("\n缺失值统计:")
    print(missing_stats[missing_stats > 0])

def save_processed_data(result, output_path='data/processed/processed_data.csv'):
    """
    保存处理后的数据
    """
    result.to_csv(output_path, index=False)
    print(f"数据已保存至: {output_path}")

def main():
    """
    主函数
    """
    # 加载并处理数据
    processed_data, missing_stats = load_and_process_data()
    
    # 打印数据摘要
    print_data_summary(processed_data, missing_stats)
    
    # 保存处理后的数据
    # save_processed_data(processed_data)
    
    return processed_data, missing_stats

if __name__ == "__main__":
    processed_data, missing_stats = main() 
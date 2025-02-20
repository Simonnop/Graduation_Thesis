import pandas as pd
import os
from datetime import datetime, timedelta
import glob

def process_huron_data(target_time=None, data_path='data/Huron/Huron_5min_merged.csv'):
    """
    处理 Huron 数据的主函数
    
    Args:
        target_time (str, optional): 目标时间点，格式为 'YYYY-MM-DD HH:MM:SS'
        data_path (str): 处理后数据的保存路径
    """
    # 读取事件数据
    event_df = pd.read_csv('data/Huron/Huron_Event_Schedule.csv')
    
    # 处理事件数据的时间格式
    event_df['timestamp'] = pd.to_datetime(event_df['Date'] + ' ' + event_df['Start Time'])
    event_df['end_timestamp'] = pd.to_datetime(event_df['Date'] + ' ' + event_df['End Time'])
    
    # 如果提供了target_time，检查是否为DR日期，如果不是则调整到最近的DR日期
    if target_time is not None:
        target_dt = pd.to_datetime(target_time)
        target_date = target_dt.date()
        
        # 获取所有DR日期
        dr_dates = pd.to_datetime(event_df['Date']).dt.date.unique()
        
        # 如果目标日期不是DR日期，找到最近的DR日期
        if target_date not in dr_dates:
            # 计算目标日期与所有DR日期的差值（绝对值）
            date_diffs = [(abs((pd.to_datetime(dr_date) - target_dt).days), dr_date) 
                         for dr_date in dr_dates]
            # 找到差值最小的日期
            closest_dr_date = min(date_diffs, key=lambda x: x[0])[1]
            
            # 保持原始时间，只改变日期
            new_target_time = pd.to_datetime(str(closest_dr_date) + ' ' + target_dt.strftime('%H:%M:%S'))
            print(f"警告：{target_time} 不是DR日期。已自动调整为最近的DR日期：{new_target_time}")
            target_time = new_target_time
    
    # 读取BAS数据
    bas_files = glob.glob('data/Huron/BAS/**/*.csv', recursive=True)
    bas_dfs = []
    
    for file in bas_files:
        try:
            df = pd.read_csv(file)
            # 合并日期和时间列
            df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            # 删除原始的日期和时间列
            df = df.drop(['Date', 'Time'], axis=1)
            # 替换"No Data"为NaN
            df = df.replace('No Data', pd.NA)
            bas_dfs.append(df)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    
    # 读取WBEL数据
    wbel_files = glob.glob('data/Huron/WBEL/**/*.csv', recursive=True)
    wbel_dfs = []
    
    for file in wbel_files:
        try:
            df = pd.read_csv(file)
            
            # 删除无名列
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
                print(f"从文件 {file} 中删除了无名列: {unnamed_cols}")
            
            # 检查文件中的列名，确定时间格式
            columns = df.columns.tolist()
            
            # 处理不同格式的时间列
            if 'Date' in columns and 'Time' in columns:
                # 如果有单独的日期和时间列
                df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df = df.drop(['Date', 'Time'], axis=1)
            elif any('Time' in col for col in columns):
                # 如果时间信息在某个列名中
                time_col = next(col for col in columns if 'Time' in col)
                try:
                    # 尝试多种日期格式
                    df['timestamp'] = pd.to_datetime(df[time_col], format='mixed', dayfirst=True)
                except:
                    print(f"警告: 无法解析时间列 {time_col} 在文件 {file} 中")
                    continue
                df = df.drop([time_col], axis=1)
            else:
                # 尝试查找第一列作为时间列
                first_col = columns[0]
                try:
                    df['timestamp'] = pd.to_datetime(df[first_col], format='mixed', dayfirst=True)
                    df = df.drop([first_col], axis=1)
                except:
                    print(f"警告: 无法在 {file} 中识别时间列")
                    print(f"列名: {columns}")
                    continue
            
            wbel_dfs.append(df)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            print(f"文件列名: {df.columns.tolist()}")
    
    # 合并所有WBEL数据并处理重复时间戳
    if wbel_dfs:
        wbel_data = pd.concat(wbel_dfs, ignore_index=True)
        # 分别处理数值列和非数值列
        numeric_cols = wbel_data.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = wbel_data.select_dtypes(exclude=['float64', 'int64']).columns.difference(['timestamp'])
        
        # 对数值列取平均值，对非数值列取最新值
        agg_dict = {col: 'mean' for col in numeric_cols}
        agg_dict.update({col: 'last' for col in non_numeric_cols})
        
        wbel_data = wbel_data.groupby('timestamp').agg(agg_dict).reset_index()
        wbel_data = wbel_data.sort_values('timestamp')
    else:
        print("没有找到有效的WBEL数据文件")
        return
    
    # 对BAS数据也进行类似处理
    if bas_dfs:
        bas_data = pd.concat(bas_dfs, ignore_index=True)
        # 分别处理数值列和非数值列
        numeric_cols = bas_data.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = bas_data.select_dtypes(exclude=['float64', 'int64']).columns.difference(['timestamp'])
        
        # 对数值列取平均值，对非数值列取最新值
        agg_dict = {col: 'mean' for col in numeric_cols}
        agg_dict.update({col: 'last' for col in non_numeric_cols})
        
        bas_data = bas_data.groupby('timestamp').agg(agg_dict).reset_index()
        bas_data = bas_data.sort_values('timestamp')
    else:
        print("没有找到有效的BAS数据文件")
        return
    
    # 创建5分钟间隔的时间序列（使用 'min' 而不是 'T'）
    start_time = min(bas_data['timestamp'].min(), wbel_data['timestamp'].min())
    end_time = max(bas_data['timestamp'].max(), wbel_data['timestamp'].max())
    time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    # 重采样数据到5分钟间隔
    processed_df = pd.DataFrame(index=time_range)
    processed_df.index.name = 'timestamp'
    
    # 重采样BAS数据
    for column in bas_data.columns:
        if column != 'timestamp':
            # 移除列名中的 HURON 前缀（包括可能的空格）
            clean_column = column.replace('HURON_', '').replace('HURON ', '')
            resampled = bas_data.set_index('timestamp')[column].reindex(time_range)
            processed_df[clean_column] = resampled.ffill()
    
    # 重采样WBEL数据
    for column in wbel_data.columns:
        if column != 'timestamp':
            # 移除列名中的 HURON 前缀（包括可能的空格）
            clean_column = column.replace('HURON_', '').replace('HURON ', '')
            resampled = wbel_data.set_index('timestamp')[column].reindex(time_range)
            processed_df[clean_column] = resampled.ffill()
    
    # 添加事件信息
    event_columns = ['Event ID', 'Type', 'Up Change', 'Down Change']
    for col in event_columns:
        processed_df[f'Event_{col}'] = None
    
    # 将事件信息添加到相应的时间点
    for _, event in event_df.iterrows():
        mask = (processed_df.index >= event['timestamp']) & (processed_df.index <= event['end_timestamp'])
        for col in event_columns:
            processed_df.loc[mask, f'Event_{col}'] = event[col]
    
    # 填充事件相关列的空值（使用适当的数据类型）
    processed_df['Event_Type'] = processed_df['Event_Type'].fillna("N")  # 字符串类型
    # 对数值列先转换类型再填充
    processed_df['Event_Up Change'] = processed_df['Event_Up Change'].astype('float64').fillna(0.0)
    processed_df['Event_Down Change'] = processed_df['Event_Down Change'].astype('float64').fillna(0.0)
    processed_df['Event_Event ID'] = processed_df['Event_Event ID'].astype('float64').fillna(0.0)
    
    # 检查并删除缺失值过多的行（每个时间点）
    # 计算每行的缺失值比例（不包括事件相关的列）
    non_event_cols = [col for col in processed_df.columns if not col.startswith('Event_')]
    row_missing_ratio = processed_df[non_event_cols].isnull().sum(axis=1) / len(non_event_cols)
    rows_to_drop = row_missing_ratio[row_missing_ratio > 0.1].index
    
    if len(rows_to_drop) > 0:
        print(f"删除缺失值过多的时间点数量: {len(rows_to_drop)}")
        print(f"删除前的数据行数: {len(processed_df)}")
        processed_df = processed_df.drop(index=rows_to_drop)
        print(f"删除后的数据行数: {len(processed_df)}")
    
    # 保存处理后的数据（更新保存路径）
    output_dir = os.path.dirname(data_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    processed_df.reset_index().to_csv(data_path, index=False)
    
    # 统计需求响应时间点数
    dr_points = (processed_df['Event_Type'] != 'N').sum()
    total_points = len(processed_df)
    dr_percentage = (dr_points / total_points) * 100
    
    print(f"处理完成，数据已保存至: {data_path}")
    print(f"需求响应时间点数: {dr_points}")
    print(f"总时间点数: {total_points}")
    print(f"需求响应时间点占比: {dr_percentage:.2f}%")

if __name__ == "__main__":
    target_time = '2021-07-27 09:00:00'
    data_path = 'data/processed/Huron_5min_merged.csv'
    process_huron_data(target_time=target_time, data_path=data_path)

# 删除缺失值过多的时间点数量: 217839
# 删除前的数据行数: 274752
# 删除后的数据行数: 56913
# 处理完成，数据已保存至: data/Huron/Huron_5min_merged.csv
# 需求响应时间点数: 1170
# 总时间点数: 56913
# 需求响应时间点占比: 2.06%
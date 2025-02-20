import pandas as pd
import os
from datetime import datetime, timedelta
import glob

def process_caspian_data():
    # 读取事件数据
    event_df = pd.read_csv('data/Caspian/Caspian_Event_Schedule.csv')
    
    # 处理事件数据的时间格式
    event_df['timestamp'] = pd.to_datetime(event_df['Date'] + ' ' + event_df['Start Time'])
    event_df['end_timestamp'] = pd.to_datetime(event_df['Date'] + ' ' + event_df['End Time'])
    
    # 读取BAS数据
    bas_files = glob.glob('data/Caspian/BAS/**/*.csv', recursive=True)
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
    wbel_files = glob.glob('data/Caspian/WBEL/**/*.csv', recursive=True)
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
            resampled = bas_data.set_index('timestamp')[column].reindex(time_range)
            processed_df[f'BAS_{column}'] = resampled.ffill()
    
    # 重采样WBEL数据
    for column in wbel_data.columns:
        if column != 'timestamp':
            resampled = wbel_data.set_index('timestamp')[column].reindex(time_range)
            processed_df[f'WBEL_{column}'] = resampled.ffill()
    
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
    
    # 保存处理后的数据
    output_path = 'data/Caspian/Caspian_5min_merged.csv'
    processed_df.reset_index().to_csv(output_path, index=False)
    print(f"处理完成，数据已保存至: {output_path}")

if __name__ == "__main__":
    process_caspian_data() 
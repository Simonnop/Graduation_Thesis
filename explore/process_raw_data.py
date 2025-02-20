import pandas as pd
import os
from datetime import datetime, timedelta
import glob

def process_data(dataset_name):
    """
    处理指定数据集的数据
    
    Args:
        dataset_name: 数据集名称（如 'Baikal', 'Huron', 'Victoria' 等）
    """
    # 读取事件数据
    try:
        event_df = pd.read_csv(f'data/raw/{dataset_name}/{dataset_name}_Event_Schedule.csv')
        # 处理事件数据的时间格式
        event_df['timestamp'] = pd.to_datetime(event_df['Date'] + ' ' + event_df['Start Time'])
        event_df['end_timestamp'] = pd.to_datetime(event_df['Date'] + ' ' + event_df['End Time'])
        has_event_data = True
    except FileNotFoundError:
        print(f"未找到{dataset_name}的事件数据文件，将创建空的事件数据")
        event_df = pd.DataFrame(columns=['timestamp', 'end_timestamp', 'Event ID', 'Type', 'Up Change', 'Down Change'])
        has_event_data = False
    
    # 读取FANPOW数据
    fanpow_files = glob.glob(f'data/raw/{dataset_name}/FANPOW/**/*.csv', recursive=True)
    fanpow_dfs = []
    
    for file in fanpow_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            # 只保留Exact列并重命名为FanPower
            if 'Exact' in df.columns:
                df = df[['Date', 'Time', 'Exact']]
                df = df.rename(columns={'Exact': 'FanPower'})
                # 合并日期和时间列
                df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                df = df.drop(['Date', 'Time'], axis=1)
                fanpow_dfs.append(df)
        except Exception as e:
            print(f"处理FANPOW文件 {file} 时出错: {str(e)}")
    
    # 读取BAS数据
    bas_files = glob.glob(f'data/raw/{dataset_name}/BAS/**/*.csv', recursive=True)
    bas_dfs = []
    
    for file in bas_files:
        try:
            df = pd.read_csv(file, low_memory=False)
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
    wbel_files = glob.glob(f'data/raw/{dataset_name}/WBEL/**/*.csv', recursive=True)
    wbel_dfs = []
    
    for file in wbel_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            
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
    
    # 处理FANPOW数据
    if fanpow_dfs:
        fanpow_data = pd.concat(fanpow_dfs, ignore_index=True)
        # 对FanPower列取平均值
        fanpow_data = fanpow_data.groupby('timestamp')['FanPower'].mean().reset_index()
        fanpow_data = fanpow_data.sort_values('timestamp')
    else:
        print("没有找到有效的FANPOW数据文件")
    
    # 更新时间范围计算
    start_time = min(bas_data['timestamp'].min(), 
                    wbel_data['timestamp'].min(),
                    fanpow_data['timestamp'].min() if len(fanpow_dfs) > 0 else pd.Timestamp.max)
    end_time = max(bas_data['timestamp'].max(), 
                  wbel_data['timestamp'].max(),
                  fanpow_data['timestamp'].max() if len(fanpow_dfs) > 0 else pd.Timestamp.min)
    
    # 创建5分钟间隔的时间序列（使用 'min' 而不是 'T'）
    time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    # 重采样数据到5分钟间隔
    processed_df = pd.DataFrame(index=time_range)
    processed_df.index.name = 'timestamp'
    
    # 重采样BAS数据
    for column in bas_data.columns:
        if column != 'timestamp':
            # 移除列名中的 BAIKAL 前缀（包括可能的空格）
            clean_column = column.replace('BAIKAL_', '').replace('BAIKAL ', '')
            resampled = bas_data.set_index('timestamp')[column].reindex(time_range)
            processed_df[clean_column] = resampled.ffill()
    
    # 重采样WBEL数据
    for column in wbel_data.columns:
        if column != 'timestamp':
            # 移除列名中的 BAIKAL 前缀（包括可能的空格）
            clean_column = column.replace('BAIKAL_', '').replace('BAIKAL ', '')
            resampled = wbel_data.set_index('timestamp')[column].reindex(time_range)
            processed_df[clean_column] = resampled.ffill()
    
    # 添加FANPOW数据的重采样
    if fanpow_dfs:
        resampled = fanpow_data.set_index('timestamp')['FanPower'].reindex(time_range)
        processed_df['FanPower'] = resampled.ffill()
    
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
    output_path = f'data/processed/{dataset_name}_5min_merged.csv'
    processed_df.reset_index().to_csv(output_path, index=False)
    
    # 统计需求响应时间点数
    dr_points = (processed_df['Event_Type'] != 'N').sum()
    total_points = len(processed_df)
    dr_percentage = (dr_points / total_points) * 100
    
    print(f"处理完成，数据已保存至: {output_path}")
    print(f"需求响应时间点数: {dr_points}")
    print(f"总时间点数: {total_points}")
    print(f"需求响应时间点占比: {dr_percentage:.2f}%")

if __name__ == "__main__":
    # 可以通过命令行参数或直接修改这里来处理不同的数据集
    datasets = ['Baikal', 'Huron', 'Victoria']
    for dataset in datasets:
        print(f"\n处理 {dataset} 数据集...")
        process_data(dataset)

# 删除缺失值过多的时间点数量: 39897
# 删除前的数据行数: 274752
# 删除后的数据行数: 234855
# 处理完成，数据已保存至: data/Baikal/Baikal_5min_merged.csv
# 需求响应时间点数: 391
# 总时间点数: 234855
# 需求响应时间点占比: 0.17%
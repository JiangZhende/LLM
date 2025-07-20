import json
import os
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
import csv
import multiprocessing as mp
from functools import partial
import time


def process_jsonl_file(file_path):
    """处理单个jsonl文件"""
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            file_lines = infile.readlines()
            
        for line in file_lines:
            json_obj = json.loads(line)
            
            prompt_text = json_obj["prompt"]
            chosen_text = json_obj["pos_resp"]
            rejected_text = json_obj["neg_resp"]
            
            data_dict = {
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text
            }
            
            processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
            lines.append(processed_line)
            
        print(f"处理完成: {file_path} - {len(lines)} 行")
        return lines
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return []


def process_parquet_file(file_path):
    """处理单个parquet文件"""
    lines = []
    try:
        df = pd.read_parquet(file_path)
        
        for idx, row in df.iterrows():
            prompt_text = row['prompt']
            chosen_text = row['chosen']
            rejected_text = row['rejected']
            
            data_dict = {
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text
            }
            
            processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
            lines.append(processed_line)
            
        print(f"处理完成: {file_path} - {len(lines)} 行")
        return lines
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return []


def process_tsv_file(file_path):
    """处理单个tsv文件"""
    lines = []
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        for idx, row in df.iterrows():
            prompt_text = row['prompt']
            chosen_text = row['chosen']
            rejected_text = row['rejected']

            data_dict = {
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text
            }
            
            processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
            lines.append(processed_line)
            
        print(f"处理完成: {file_path} - {len(lines)} 行")
        return lines
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return []


def merge_datasets_parallel(input_dir, num_processes=None):
    """多进程版本的数据集合并函数"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    # 收集所有需要处理的文件
    jsonl_files = []
    parquet_files = []
    tsv_files = []
    
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith('.jsonl'):
                jsonl_files.append(file_path)
            elif file.endswith('.parquet'):
                parquet_files.append(file_path)
            elif file.endswith('.tsv'):
                tsv_files.append(file_path)
    
    print(f"找到 {len(jsonl_files)} 个jsonl文件, {len(parquet_files)} 个parquet文件, {len(tsv_files)} 个tsv文件")
    
    total_lines = []
    
    # 使用进程池并行处理文件
    with mp.Pool(processes=num_processes) as pool:
        # 处理jsonl文件
        if jsonl_files:
            print("开始处理jsonl文件...")
            jsonl_results = list(tqdm(
                pool.imap(process_jsonl_file, jsonl_files),
                total=len(jsonl_files),
                desc="处理jsonl文件"
            ))
            for result in jsonl_results:
                total_lines.extend(result)
        
        # 处理parquet文件
        if parquet_files:
            print("开始处理parquet文件...")
            parquet_results = list(tqdm(
                pool.imap(process_parquet_file, parquet_files),
                total=len(parquet_files),
                desc="处理parquet文件"
            ))
            for result in parquet_results:
                total_lines.extend(result)
        
        # 处理tsv文件
        if tsv_files:
            print("开始处理tsv文件...")
            tsv_results = list(tqdm(
                pool.imap(process_tsv_file, tsv_files),
                total=len(tsv_files),
                desc="处理tsv文件"
            ))
            for result in tsv_results:
                total_lines.extend(result)
    
    print(f"总共处理了 {len(total_lines)} 行数据")
    
    # 如果输出子文件夹不存在，则创建它
    output_subfolder = "datasets/rlhf"
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # 保存处理后的数据到jsonl文件
    output_file_path = os.path.join(output_subfolder, "rl_data.jsonl")
    print(f"正在保存数据到 {output_file_path}...")
    
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in total_lines:
            outfile.write(line)
    
    print(f"数据保存完成，共 {len(total_lines)} 行")


def merge_datasets(input_dir):
    """原始单进程版本（保留作为备用）"""
    total_lines = []
    for subdir, dirs, files in os.walk(input_dir):
        for idx, file in enumerate(files):
            # 只处理txt文件
            if file.endswith('.jsonl'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                print(file_path)
                # 读取jsonl文件
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                    
                for line in tqdm(lines):
                    json_obj = json.loads(line)  # 解析json字符串为python对象
                    
                    prompt_text = json_obj["prompt"]
                    chosen_text = json_obj["pos_resp"]
                    rejected_text = json_obj["neg_resp"]
                    
                    data_dict = {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }
                    
                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
                    total_lines.append(processed_line)

            if file.endswith('.parquet'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                print(file_path)
                # 读取jsonl文件
                df = pd.read_parquet(file_path)
                
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    prompt_text = row['prompt']
                    chosen_text = row['chosen']
                    rejected_text = row['rejected']
                    
                    data_dict = {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }
                    
                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
                    total_lines.append(processed_line)
    
            if file.endswith('.tsv'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                print(file_path)
                # 读取jsonl文件
                df = pd.read_csv(file_path, sep='\t')
                
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    prompt_text = row['prompt']
                    chosen_text = row['chosen']
                    rejected_text = row['rejected']

                    data_dict = {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }
                    
                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n' 
                    total_lines.append(processed_line)
                    
    # 如果输出子文件夹不存在，则创建它
    output_subfolder = "data/rl_train"
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # 保存处理后的csv文件到对应的输出子文件夹
    output_file_path = os.path.join(output_subfolder, "rl_data.jsonl")
    # 将处理后的json对象写入新的jsonl文件
    with open(output_file_path, 'w') as outfile:
        for line in total_lines:
            outfile.write(line)


if __name__=="__main__":
    # 使用多进程版本
    start_time = time.time()
    merge_datasets_parallel("datasets/rlhf")
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")
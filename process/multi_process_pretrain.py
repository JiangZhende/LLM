import json
import os
import glob
from traceback import format_exc
import numpy as np
from tqdm import tqdm
from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
import concurrent.futures

def chunk_list(lst, chunk_size):
    """按固定大小分割列表"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i: i+chunk_size]

def process_and_write_batch(lines, tokenizer_path, process_fun, output_file_path, num_workers):
    """处理一批文本，并将结果写入输出文件"""
    if not lines:
        return

    line_batches = list(chunk_list(lines, max(1, len(lines) // num_workers)))

    all_tokens = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_fun, batch, tokenizer_path) for batch in line_batches]
        for future in concurrent.futures.as_completed(futures):
            try:
                tokens = future.result()
                all_tokens.extend(tokens)
            except Exception as e:
                print(f"[Warning] Error in worker: {e}")

    arr = np.array(all_tokens, dtype=np.uint16)
    with open(output_file_path, "ab") as f:
        f.write(arr.tobytes())
        
def process_file(file_path, tokenizer_path, process_line, batch_size=10000, max_workers=4):
    base_name, _ = os.path.splitext(file_path)
    output_file_path = base_name + ".bin"
    # 确保文件是空的
    open(output_file_path, "wb").close()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
        batch_lines = []
        batch = []
        pbar = tqdm(desc=f"processing {os.path.basename(file_path)}")

        for line in infile:
            batch.append(line)
            if len(batch) >= batch_size:
                process_and_write_batch(batch, tokenizer_path, process_line, output_file_path, max_workers)
                pbar.update(len(batch))
                batch.clear()
        if batch:
            process_and_write_batch(batch, tokenizer_path, process_line, output_file_path, max_workers)
            pbar.update(len(batch))

        pbar.close()
    return output_file_path

def process_wiki(input_dir, tokenizer_path, batch_size=10000, max_workers=4):
    tokenizer_path = tokenizer.vocab_file
    file_list = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                file_list.append(file_path)

    for file_path in file_list:
        print(f"processing: {file_path}")
        process_file(file_path, tokenizer_path, process_wiki_line, batch_size, max_workers)

def process_wiki_file(file_path, tokenizer_path, batch_size=10000, max_workers=4):
    base_name, _ = os.path.splitext(file_path)
    output_file_path = base_name + ".bin"
    # 确保文件是空的
    open(output_file_path, "wb").close()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
        batch_lines = []
        batch = []
        pbar = tqdm(desc=f"processing {os.path.basename(file_path)}")

        for line in infile:
            batch.append(line)
            if len(batch) >= batch_size:
                process_and_write_batch(batch, tokenizer_path, process_wiki_line, output_file_path, max_workers)
                pbar.update(len(batch))
                batch.clear()
        if batch:
            process_and_write_batch(batch, tokenizer_path, output_file_path, max_workers)
            pbar.update(len(batch))

        pbar.close()
    return output_file_path

def process_wiki_line(lines, tokenizer_path):
    from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)

    tokens_all = []
    for line in lines:
        try:
            json_obj = json.loads(line)
            text = json_obj.get("completion", "")
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens["<eos>"])
            if len(tokens) > 5:
                tokens_all.extend(tokens)
        except Exception as e:
            print(f"Error: {e}")
    return tokens_all


def process_webnovel_line(lines, tokenizer_path):
    from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)

    tokens_all = []
    for line in lines:
        try:
            json_obj = json.loads(line)
            text = json_obj.get("text", "")
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens["<eos>"])
            if len(tokens) > 5:
                tokens_all.extend(tokens)
        except Exception as e:
            print(f"Error: {e}")
    return tokens_all


def process_webnovel_file(file_path, tokenizer_path, batch_size=100000, num_workers=4):
    base_name, _ = os.path.splitext(file_path)
    output_file_path = base_name + ".bin"
    # 确保文件是空的
    open(output_file_path, "wb").close()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
        batch_lines = []
        batch = []
        pbar = tqdm(desc=f"processing {os.path.basename(file_path)}")

        for line in infile:
            batch.append(line)
            if len(batch) >= batch_size:
                process_and_write_batch(batch, tokenizer_path, process_webnovel_line, output_file_path, num_workers)
                pbar.update(len(batch))
                batch.clear()
        if batch:
            process_and_write_batch(batch, tokenizer_path, output_file_path, num_workers)
            pbar.update(len(batch))

        pbar.close()
    return output_file_path


def  process_webnovel(input_dir, tokenizer_path, batch_size, max_workers):
    # tokenizer_path = tokenizer.vocab_file
    file_list = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir, file)
                file_list.append(file_path)

    for file_path in file_list:
        print(f"processing: {file_path}")
        process_file(file_path, tokenizer_path, process_webnovel_line, batch_size, max_workers)
        # break



def process_tigerbo_file(file_path, tokenizer_path, batch_size=10000, num_workers=4):
    base_name, _ = os.path.splitext(file_path)
    output_file_path = base_name + ".bin"

    # 确保文件是空的
    open(output_file_path, "wb").close()
    import pyarrow.parquet as pq
    
    # with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
    pf = pq.ParquetFile(file_path)
    pbar = tqdm(desc=f"processing: {os.path.basename(file_path)}")
    for lines in pf.iter_batches(batch_size=batch_size):
        batch = []
        df = lines.to_pandas()
        df = df[df["content"].notna()]
        df = df[df["content"].str.strip() != ""]
        df = df[df["content"].str.len() > 5]
        df = df[df["content"].str.len() < 10000]
        df = df.reset_index(drop=True)
        for _, row in df.iterrows():
            batch.append(row["content"])
        process_and_write_batch(batch, tokenizer_path, process_zhihu_line, output_file_path, num_workers)
        pbar.update(len(batch))
        batch.clear()
    pbar.close()
    return output_file_path
                

def process_tigerbot(input_dir, tokenizer_path, batch_size, max_workers):
    file_list = glob.glob(os.path.join(input_dir, "*.parquet"))
    for file_path in file_list: 
        process_tigerbo_file(file_path, tokenizer_path, batch_size, max_workers)


def process_zhihu_line(lines, tokenizer_path):
    from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    import numpy as np
    import pandas as pd
    import os
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
    all_tokens = []
    tokens_all = []
    for text in lines:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens["<eos>"])
            if len(tokens) > 5:
                tokens_all.extend(tokens)
        except Exception as e:
            print(f"Error: {format_exc()}")
    return tokens_all


def process_zhihu_file(file_path, tokenizer_path, batch_size=10000, num_workers=4):
    base_name, _ = os.path.splitext(file_path)
    output_file_path = base_name + ".bin"

    # 确保文件是空的
    open(output_file_path, "wb").close()

    # with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
    pf = pd.read_parquet(file_path)
    if "RESPONSE" not in pf.columns:
        raise ValueError(f"File {file_path} does not contain 'RESPONSE' column.")
    pf = pf[pf["RESPONSE"].notna()]
    pf = pf[pf["RESPONSE"].str.strip() != ""]
    pf = pf[pf["RESPONSE"].str.len() > 5]
    pf = pf[pf["RESPONSE"].str.len() < 10000]
    pf = pf.reset_index(drop=True)
    batch = []
    pbar = tqdm(desc=f"processing: {os.path.basename(file_path)}")
    for _, row in pf.iterrows():
        batch.append(row["RESPONSE"])
        if len(batch) >= batch_size:
            process_and_write_batch(batch, tokenizer_path, process_zhihu_line, output_file_path, num_workers)
            pbar.update(len(batch))
            batch.clear()
    if batch:
        process_and_write_batch(batch, tokenizer_path, process_zhihu_line, output_file_path, num_workers)
        pbar.update(len(batch))
    pbar.close()
    return output_file_path

def process_zhihu(input_dir, tokenizer_path, batch_size, max_workers):
    # tokenizer_path = tokenizer.vocab_file
    file_list = glob.glob(os.path.join(input_dir, "*.parquet"))
    for file_path in file_list:
        process_zhihu_file(file_path, tokenizer_path, batch_size, max_workers)


def process_baidubaike_line(lines, tokenizer_path):
    from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    import numpy as np
    import pandas as pd
    import os
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
    all_tokens = []
    tokens_all = []
    for line in lines:
        text = ""
        try:
            json_obj = json.loads(line)
            # print(json_obj)
            summary = json_obj["summary"] if json_obj["summary"] else ""
            text += json_obj.get("title", "") + ": " + summary
            for section in json_obj.get("sections", []):
                text += section.get("title", "") + ": " + section.get("content", "")
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens["<eos>"])
            if len(tokens) > 5:
                tokens_all.extend(tokens)
        except Exception as e:
            print(f"Error: {format_exc()}")
    return tokens_all

def process_baidubaike_file(file_path, tokenizer_path, batch_size=10000, max_workers=4):
    base_name, _ = os.path.splitext(file_path)
    output_file_path = base_name + ".bin"

    # 确保文件是空的
    open(output_file_path, "wb").close()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
        batch = []
        pbar = tqdm(desc=f"processing: {os.path.basename(file_path)}")
        for line in infile:
            batch.append(line)
            if len(batch) >= batch_size:
                process_and_write_batch(batch, tokenizer_path, process_baidubaike_line, output_file_path, max_workers)
                pbar.update(len(batch))
                batch.clear()
        if batch:
            process_and_write_batch(batch, tokenizer_path, output_file_path, max_workers)
            pbar.update(len(batch))
        pbar.close()
    return output_file_path

def process_baidubaike(input_dir, tokenizer_path, batch_size=10000, max_workers=4):
    file_list = glob.glob(os.path.join(input_dir, "*.json"))
    for file_path in file_list:
        process_file(file_path, tokenizer_path, process_baidubaike_line, batch_size, max_workers)
    

if __name__ == "__main__":
    tokenizer = ChatGLMTokenizer(vocab_file="glm3_tokenizer/tokenizer.model")
    # 示例调用
    # process_wiki_clean("datasets/pretrain/pleisto/wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json", tokenizer)
    # process_webnovel("datasets/pretrain/wdndev/webnovel-chinese/data", "glm3_tokenizer/tokenizer.model", 100000, 6)
    # process_zhihu("datasets/pretrain/wangrui6/Zhihu-KOL/data", "glm3_tokenizer/tokenizer.model", 100000, 6)
    # process_tigerbot("datasets/pretrain/TigerResearch/pretrain_zh/data", "glm3_tokenizer/tokenizer.model", 100000, 6)
    # process_baidubaike("/root/LLM/datasets/pretrain/xuqinyang/BaiduBaike-5.63M/", "glm3_tokenizer/tokenizer.model", 100000, 6) 

    process_tigerbo_file("datasets/pretrain/TigerResearch/pretrain_zh/data/train-00006-of-00117-60d762f229f705bc.parquet",
                          "glm3_tokenizer/tokenizer.model", 10000, 6)
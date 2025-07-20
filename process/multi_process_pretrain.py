import json
import os
import glob
import numpy as np
from tqdm import tqdm
from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
import concurrent.futures

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

def process_webnovel_file(file_path, tokenizer_path, batch_size=10000):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
        lines = infile.readlines()
    def chunk_list(lst, chunk_size):
        """按固定大小分割列表"""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]
    
    line_batches = list(chunk_list(lines, batch_size))
    all_tokens = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_webnovel_line, batch, tokenizer_path)
            for batch in line_batches
        ]        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(file_path)}"):
            tokens = future.result()
            all_tokens.extend(tokens)

    arr = np.array(all_tokens, dtype=np.uint16)
    base_name, ext = os.path.splitext(file_path)
    output_file_path = base_name + ".bin"
    with open(output_file_path, "wb") as f:
        f.write(arr.tobytes())
    return output_file_path

def process_webnovel(input_dir, tokenizer):
    tokenizer_path = tokenizer.vocab_file
    file_list = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir, file)
                file_list.append(file_path)

    for file_path in file_list:
        process_webnovel_file(file_path, tokenizer_path)

def process_tigerbot_wiki_file(file_path, tokenizer_path):
    from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    import numpy as np
    import os
    import json
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
    all_tokens = []
    with open(file_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
    for line in lines:
        json_obj = json.loads(line)
        text = json_obj["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.special_tokens["<eos>"])
        if len(tokens) > 5:
            all_tokens += tokens
    arr = np.array(all_tokens, dtype = np.uint16)
    base_name, ext = os.path.splitext(file_path)
    output_file_path = base_name + ".bin"
    with open(output_file_path, "wb") as f:
        f.write(arr.tobytes())
    return output_file_path

def process_tigerbot_wiki(input_dir, tokenizer):
    tokenizer_path = tokenizer.vocab_file
    file_list = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                file_list.append(file_path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_tigerbot_wiki_file, file_path, tokenizer_path) for file_path in file_list]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="process_tigerbot_wiki"):
            _ = f.result()

def process_tigerbot_part_file(file, tokenizer_path):
    from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    import numpy as np
    import pandas as pd
    import os
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
    all_tokens = []
    df = pd.read_parquet(file)
    responses = df["content"]
    for text in responses:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.special_tokens["<eos>"])
        if len(tokens) > 5:
            all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    base_name, ext = os.path.splitext(file)
    output_file_path = base_name + ".bin"
    with open(output_file_path, "wb") as f:
        f.write(arr.tobytes())
    return output_file_path

def process_tigerbot_part(input_dir, tokenizer):
    tokenizer_path = tokenizer.vocab_file
    file_list = glob.glob(os.path.join(input_dir, "*.parquet"))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_tigerbot_part_file, file, tokenizer_path) for file in file_list]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="process_tigerbot_part"):
            _ = f.result()

def process_zhihu_file(file, tokenizer_path):
    from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    import numpy as np
    import pandas as pd
    import os
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
    all_tokens = []
    df = pd.read_parquet(file)
    responses = df["RESPONSE"]
    for text in responses:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.special_tokens["<eos>"])
        if len(tokens) > 5:
            all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    base_name, ext = os.path.splitext(file)
    output_file_path = base_name + ".bin"
    with open(output_file_path, "wb") as f:
        f.write(arr.tobytes())
    return output_file_path

def process_zhihu(input_dir, tokenizer):
    tokenizer_path = tokenizer.vocab_file
    file_list = glob.glob(os.path.join(input_dir, "*.parquet"))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_zhihu_file, file, tokenizer_path) for file in file_list]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="process_zhihu"):
            _ = f.result()

def process_wiki_clean_chunk(chunk_data, chunk_idx, tokenizer_path, output_dir):
    from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    import numpy as np
    import os
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
    all_tokens = []
    for line in chunk_data:
        text = line["completion"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.special_tokens['<eos>'])
        if len(tokens) > 5:
            all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    output_file_path = os.path.join(output_dir, f"wiki_clean_chunk_{chunk_idx}.bin")
    with open(output_file_path, 'wb') as f:
        f.write(arr.tobytes())
    return output_file_path

def process_wiki_clean(file_path, tokenizer, chunk_size=25000):
    import math
    import shutil
    base_name, ext = os.path.splitext(file_path)
    output_file_path = base_name + '.bin'
    output_dir = os.path.dirname(file_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    num_chunks = math.ceil(total / chunk_size)
    tokenizer_path = tokenizer.vocab_file
    chunk_paths = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(num_chunks):
            chunk_data = data[i*chunk_size:(i+1)*chunk_size]
            futures.append(executor.submit(process_wiki_clean_chunk, chunk_data, i, tokenizer_path, output_dir))
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="process_wiki_clean_multi"):
            chunk_paths.append(f.result())
    # 合并所有bin文件
    chunk_paths = [os.path.join(output_dir, f"wiki_clean_chunk_{i}.bin") for i in range(num_chunks)]
    arrs = []
    for p in chunk_paths:
        with open(p, "rb") as f:
            arr = np.fromfile(f, dtype=np.uint16)
            arrs.append(arr)
    arr = np.concatenate(arrs)
    # base_name, ext = os.path.splitext(os.path.basename(file_path))
    # output_file_path = os.path.join(output_dir, base_name + ".bin")
    with open(output_file_path, "wb") as f:
        f.write(arr.tobytes())
    # 可选：删除临时分块bin文件
    for p in chunk_paths:
        os.remove(p)
    print(f"合并完成，输出文件: {output_file_path}")

def process_baidu_baike_batch(lines, tokenizer_path, batch_index):
    tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
    doc_ids = []
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception as e:
            print(f"Error decoding JSON: {e}")
            continue

        text = ""
        try:
            text += obj.get("title", "") + ": " + obj.get("summary", "")
        except Exception:
            pass
        for section in obj.get("sections", []):
            try:
                text += section.get("title", "") + ": " + section.get("content", "")
            except Exception:
                pass

        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens["<eos>"])
        if len(text_id) > 5:
            doc_ids.extend(text_id)

    arr = np.array(doc_ids, dtype=np.uint16)
    output_file = f"./baidubaike_5632_{batch_index}.bin"
    with open(output_file, "wb") as f:
        f.write(arr.tobytes())
    print(f"Batch {batch_index} processed with {len(doc_ids)} tokens, saved to {output_file}")
    return output_file


def process_baidu_baike_mp(file_path, tokenizer_path, batch_size=10000, max_workers=4):
    with open(file_path, "r", encoding="utf-8") as f:
        lines_buffer = []
        batch_index = 0
        futures = []
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for line in f:
                lines_buffer.append(line)
                if len(lines_buffer) >= batch_size:
                    batch_index += 1
                    futures.append(executor.submit(process_baidu_baike_batch, lines_buffer, tokenizer_path, batch_index))
                    lines_buffer = []

            # 处理剩余不足 batch_size 的数据
            if lines_buffer:
                batch_index += 1
                futures.append(executor.submit(process_baidu_baike_batch, lines_buffer, tokenizer_path, batch_index))

            # 等待所有任务完成
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    return results

if __name__ == "__main__":
    tokenizer = ChatGLMTokenizer(vocab_file="glm3_tokenizer/tokenizer.model")
    # 示例调用
    # process_wiki_clean("datasets/pretrain/pleisto/wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json", tokenizer)
    # process_webnovel("datasets/pretrain/wdndev/webnovel-chinese/data", tokenizer)
    process_zhihu("datasets/pretrain/wangrui6/Zhihu-KOL/data", tokenizer)
    # process_tigerbot_part("datasets/TigerResearch/pretrain_zh", tokenizer)
    process_baidu_baike_mp("/root/LLM/datasets/pretrain/xuqinyang/BaiduBaike-5.63M/563w_baidubaike.json", tokenizer) 
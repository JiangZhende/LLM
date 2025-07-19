import json
import os
import glob
import numpy as np
from tqdm import tqdm
from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
import concurrent.futures

def process_webnovel_file(file_path, tokenizer_path):
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

def process_webnovel(input_dir, tokenizer):
    tokenizer_path = tokenizer.vocab_file
    file_list = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir, file)
                file_list.append(file_path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_webnovel_file, file_path, tokenizer_path) for file_path in file_list]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="process_webnovel"):
            _ = f.result()

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

def process_wiki_clean_multi(file_path, tokenizer, chunk_size=25000, output_dir="wiki_clean_chunks"):
    import math
    import shutil
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
    base_name, ext = os.path.splitext(os.path.basename(file_path))
    output_file_path = os.path.join(output_dir, base_name + ".bin")
    with open(output_file_path, "wb") as f:
        f.write(arr.tobytes())
    # 可选：删除临时分块bin文件
    for p in chunk_paths:
        os.remove(p)
    print(f"合并完成，输出文件: {output_file_path}")

if __name__ == "__main__":
    tokenizer = ChatGLMTokenizer(vocab_file="glm3_tokenizer/tokenizer.model")
    # 示例调用
    # process_webnovel("datasets/webnovel-chinese/data", tokenizer)
    # process_zhihu("datasets/wangrui6/Zhihu-KOL/data", tokenizer)
    # process_tigerbot_part("datasets/TigerResearch/pretrain_zh", tokenizer)
    # process_tigerbot_wiki("datasets/tigerbot/wiki", tokenizer) 
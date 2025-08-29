import json
import os
import glob
import numpy as np
from tqdm import tqdm
from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
import concurrent.futures

def process_wiki_clean(file_path, tokenizer):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_tokens = []
    for line in tqdm(data):
        text = line["completion"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.special_tokens['<eos>'])
        if len(tokens) > 5:
            all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    base_name, ext = os.path.splitext(file_path)
    output_file_path = base_name + '.bin'
    with open(output_file_path, 'wb') as f:
        f.write(arr.tobytes())
        

def process_webnovel(input_dir, tokenizer):
    for subdir, dirs, files in os.walk(input_dir):
        for idx, file in enumerate(files):
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir, file)
                all_tokens = []

                with open(file_path, "r", encoding="utf-8") as infile:
                    lines = infile.readlines()
                
                for line in tqdm(lines):
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


def process_tigerbot_wiki(input_dir, tokenizer):
    for subdir, dirs, files in os.walk(input_dir):
        for idx, file in enumerate(files):
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                all_tokens = []
                with open(file_path, "r", encoding="utf-8") as infile:
                    lines = infile.readlines()

                for line in tqdm(lines):
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


def process_tigerbot_part(input_dir, tokenizer):
    all_tokens = []
    total_len = 0
    file_idx = 7
    # print(os.listdir(input_dir))
    for file in glob.glob(os.path.join(input_dir, "*.parquet")):
        print(file)
        df = pd.read_parquet(file)
        responses = df["content"]
        for text in tqdm(responses):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens["<eos>"])
            if len(tokens) > 5:
                all_tokens += tokens
        total_len += len(df)
        if total_len > 600000:
            arr = np.array(all_tokens, dtype=np.uint16)
            output_file_path = "tigerbot_part_" + str(file_idx) + ".bin"
            with open(output_file_path, "wb") as f:
                f.write(arr.tobytes())
            all_tokens = []
            total_len = 0
            file_idx += 1
    if len(all_tokens) > 0:
        arr = np.array(all_tokens, dtype=np.uint16)
        output_file_path = "tigerbot_part_" + str(file_idx) + ".bin"
        with open(output_file_path, "wb") as f:
            f.write(arr.tobytes())


def process_zhihu(input_dir, tokenizer):
    all_tokens = []
    for file in glob.glob(os.path.join(input_dir, "*.parquet")):
        print(file)
        df = pd.read_parquet(file)
        responses = df["RESPONSE"]
        for text in tqdm(responses):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens["<eos>"])
            if len(tokens) > 5:
                all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    output_file_path = "zhihu" + ".bin"
    with open(output_file_path, "wb") as f:
        f.write(arr.tobytes())

def process_baidu_baike(input_dir, tokenizer):
    BATCH_SIZE = 1000000
    cnt = 0
    batch_cnt = 0
    token = 0
    doc_ids = []
    f1 = open(input_dir, "r", encoding="utf-8")
    while True:
        line = f1.readline()
        if not line:
            break
        line = json.loads(line)
        text = ""
        try:
            text  += line["title"] + ": " + line["summary"]
        except:
            pass
        for per in line["sections"]:
            text += per["title"] + ": " + line["content"]
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens["<eos>"])
        if len(text_id) > 5:
            doc_ids += text_id
        cnt += 1
        if cnt % BATCH_SIZE == 0:
            batch_cnt += 1
            arr = np.array(doc_ids, dtype=np.uint16)
            doc_ids = []
            print("cnt:", cnt, "arr_shape:", arr.shape)
            with open("./baidubaike_5632_{}.bin".format(batch_cnt), "wb") as f:
                f.write(arr.tobytes())
            del arr

        if not doc_ids:
            batch_cnt += 1
            arr = np.array(doc_ids, dtype=np.uint16)
            print("cnt:", cnt, "arr_shape:", arr.shape)
            with open("./baidubaike_5632_{}.bin".format(batch_cnt), "wb") as f:
                f.write(arr.tobytes())


def merge_bin(data_path_list: list):
    data_arr = []
    for data_path in tqdm(data_path_list):
        with open(data_path, "rb") as f:
            data = np.fromfile(f, dtype = np.uint16)
            data_arr.append(data)
    arr = np.concatenate(data_arr)
    print(arr.shape)
    
    with open("./data/pretrain_data.bin", "wb") as f:
        f.write(arr.tobytes())
if __name__ == "__main__":
    tokenizer = ChatGLMTokenizer(vocab_file="glm3_tokenizer/tokenizer.model")
    # process_wiki_clean("datasets/pretrain/pleisto/wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json", tokenizer)
    # process_webnovel("datasets/webnovel-chinese/data", tokenizer)
    # process_zhihu("datasets/wangrui6/Zhihu-KOL/data", tokenizer)
    process_tigerbot_part("datasets/pretrain/TigerResearch/pretrain_zh/data", tokenizer)
    # process_baidu_baike("datasets/xuqinyang/BaiduBaike-5.63M", tokenizer)
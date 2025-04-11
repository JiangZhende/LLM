import json
import os
import glob
import numpy as np
from tqdm import tqdm
from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd

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
        

if __name__ == "__main__":
    tokenizer = ChatGLMTokenizer(vocab_file="/Users/likun/code/LLM/glm3_tokenizer/tokenizer.model")
    process_wiki_clean("/Users/likun/code/LLM/dataset/wikipedia-cn-20230720-filtered.json", tokenizer)
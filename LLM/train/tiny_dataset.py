import numpy as np
import random
import jsonlines

import torch
from torch.utils.data import Dataset
from typing import Dict

class PTMDataset(Dataset):
    def __init__(self, data_path_list, max_length=512, ) -> None:
        super(PTMDataset, self).__init__()
        data_list = []
        for data_path in data_path_list:
            with open(data_path, "rb") as f:
                data = np.fromfile(f, dtype=np.uint16)
                data_list.append(data)
        data = np.concatenate(data_list)
        data = data[:max_length * int(len(data) / max_length)]
        self.data = data.reshape(-1, max_length)
        self.shuffle_index = list(range(len(self.data)))
        random.shuffle(self.shuffle_index)
        print("Data loading is completed.")
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index: int):
        index = self.shuffle_index[index]
        sample = self.data[index]
        X = np.array(sample).astype(np.int64)
        input_ids = torch.LongTensor(X)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }
    

class SFTDataset(Dataset):
    def __init__(
            self,
            data_path,
            tokenizer,
            max_length=256,
            system: str = "你是由李小贱开发的个人助手。"
    ):
        super(SFTDataset, self).__init__()
        self.data = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                self.data.append(obj)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system = system
        

    def preprocessing(self, example, debug=False):
        input_ids, labels = [], []
        prompt_txt = self.system
        user_txt = example["question"]
        assistant_txt = example["answer"]

        instruction = self.tokenizer.encode(text="\n".join(["<|system|>", prompt_txt.strip(),
                                                            "<|user|>", user_txt.strip(),
                                                            "<|assistant|>"]).strip()+"\n",
                                            add_special_tokens=True,
                                            truncation=True,
                                            max_length=self.max_length)
        response = self.tokenizer.encode(assistant_txt.strip(), add_special_tokens=False,
                                         truncation=True, max_length=self.max_length)
        input_ids = instruction + response + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * len(instruction) + response + [self.tokenizer.eos_token_id]
        if (len(input_ids) > self.max_length):
            return None
        if debug:
            print(self.tokenizer.decode(input_ids))
            print("--------------------------")
        
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        processed_example = self.preprocessing(self.data[index])
        while processed_example is None:
            index = (index + 1) % len(self.data)
            processed_example = self.preprocessing(self.data[index])

        return processed_example
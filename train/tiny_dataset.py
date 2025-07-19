import numpy as np
import random
import jsonlines

import torch
from torch.utils.data import Dataset
from typing import Dict
import datasets
from datasets import load_dataset

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


def load_dpo_dataset(
    data_path: str,
    max_length=256,
    sanity_check: bool = False,
    num_proc=24,
    system: str = "你是由wdndev开发的个人助手。",
) -> datasets.Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        'json',
        data_files=data_path,
        split="train",
        # cache_dir=cache_dir,
        # data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def preprocess_function(examples) -> Dict[str, str]:
        prompt_txt = system
        # assistant_chosen_txt = examples["chosen"]
        # assistant_rejected_txt = examples["rejected"]

        prompt_list = []
        for question in examples["prompt"]:
            prompt ="\n".join(["<|system|>", system.strip(), 
                            "<|user|>", question.strip(), 
                            "<|assistant|>"]).strip() + "\n"
            prompt_list.append(prompt)

        return {
            "prompt": prompt_list,
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
        }
        
        # return {
        #     "prompt": ["Question: " + question + "\n\nAnswer: " for question in examples["prompt"]],
        #     "chosen": examples["chosen"],
        #     "rejected": examples["rejected"],
        # }

    dataset_map = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    dataset_map = dataset_map.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length
        and len(x["prompt"]) + len(x["rejected"]) <= max_length
    )

    # dataset_map.set_format(type="torch")

    return dataset_map
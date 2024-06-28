import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('/ssd1/liboran/gitspace/github_Librarvl/LLM-experiment/')

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Trainer, TrainingArguments


import pandas as pd
from datasets import load_dataset
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset

from transformers_s.src.transformers.models.llama import *


def tf_load_dataset():
    data = load_dataset("dataset/rbt3/")
    tokenizer = AutoTokenizer.from_pretrained("dataset/rbt3/")
    data = 1
    print(data)

    return data


def load_model():
    # data_set = tf_load_dataset()
    path = "/home/work/wenku_yq/DataVault/models/Meta-Llama-3-8B"
    path = "/home/work/wenku_yq/DataVault/models/Llama2-7b-hf"
    llama_config = AutoConfig.from_pretrained(path)
    # llama_config = LlamaConfig()
    llama_model = LlamaModel.from_pretrained(path)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # llama_model

    return llama_model



def main():
    load_model()


if __name__ == "__main__":
    main()
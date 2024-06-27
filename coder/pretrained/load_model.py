import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('/ssd1/liboran/gitspace/github_Librarvl/LLM-experiment/')

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    data_set = tf_load_dataset()

    llama_config = LlamaConfig()
    llama_model = LlamaModel(llama_config)

    a = 1

    # llama_model

    return llama_model



def main():
    load_model()


if __name__ == "__main__":
    main()
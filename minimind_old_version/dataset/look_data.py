import json
from transformers import AutoTokenizer




def eval_tokenizer(data):
    tokenizer = AutoTokenizer.from_pretrained("/home/boran.lbr/gitspace/github/minimind/model")
    max_length = 10

    for messages in data:
        messages = json.loads(messages)["text"]
        model_inputs = tokenizer(
            messages,
            # max_length=max_length,
            # padding='max_length',
            # truncation=True,
            return_tensors='pt'
        )

        if model_inputs['input_ids'].shape[1] > 512:
            print('encoder长度：', model_inputs['input_ids'].shape[1])


def main():
    data = []
    with open("/home/boran.lbr/gitspace/github/minimind/dataset/minimind_dataset/pretrain_hq.jsonl", "r") as f:
        for item in f.readlines():
            data.append(item)
    
    eval_tokenizer(data)


if __name__ == '__main__':
    main()
# 安装 llamafactory-cli

pip install -e .[metrics]

pip install -e '.[torch-npu,metrics]'

# 查看显存

npu-smi info

# 安装 vllm

pip install synr==0.5.0

pip install tornado

pip install ydantic=2.0.0

pip install -e vllm

# 写 yaml

## fastapi

model_name_or_path: /home/work/wenku_yq/DataVault/models/Yi-1.5-34B-Chat

template: yi

## vllm

补充：

infer_backend: vllm

vllm_enforce_eager: true

# 运行

API_PORT=8134 llamafactory-cli api ./inference/Yi1.5_34B.yaml

qwen2 72B   Yi 1.5  34B  Llama3 70B    CR+

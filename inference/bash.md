# 启动vllm api服务

## word2ppt - api2 - Qwen1.5-72B

python -m vllm.entrypoints.openai.api_server --model ./afs/Qwen1.5-72B/word2ppt_api2 --gpu-memory-utilization 0.95 --tensor-parallel-size 8 --max-model-len 32768 --port 8888

python -m vllm.entrypoints.openai.api_server --model ./afs/checkpoint_72B_chat/word2ppt_api1 --gpu-memory-utilization 0.95 --tensor-parallel-size 8 --max-model-len 32768 --port 8888

# vllm 环境

/home/users/hanwanpeng/.config/vllm/nccl/cu11/libnccl.so.2.18.1

# [RayletClient] Unable to register worker with raylet. Unknown error

pip install ray==2.5.1

pip install grpcio==1.51.3


# LLaMA Factory 启动 vllm

API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml

export CUDA_VISIBLE_DEVICES=4,5,6,7

CUDA_VISIBLE_DEVICES=0,1,2,3 API_PORT=8889 llamafactory-cli api examples/inference/qwen2_72b_vllm.yaml

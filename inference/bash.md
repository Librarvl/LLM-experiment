
# 启动vllm api服务

## word2ppt - api2 - Qwen1.5-72B
python -m vllm.entrypoints.openai.api_server --model ./afs/Qwen1.5-72B/word2ppt_api2 --gpu-memory-utilization 0.95 --tensor-parallel-size 8 --max-model-len 32768 --port 8888
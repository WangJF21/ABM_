import os
from pathlib import Path
class Config:
    model_name = "deepseek-v4-flash"
    api_key = "sk-8f6f9069c25f48b8b4eb5325b1da253a"
    url = "https://api.deepseek.com/v1/"
    max_tokens = 2048
    temperature = 0.4
    embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim = 1024
    embedding_url = "https://api.siliconflow.cn/v1/embeddings"
    embedding_api_key = "sk-zoirtejerfekrngpaawphdvmqjkbuqghmkxcnragodzlkynh"
    data_path = Path("data/individual_simulation_data")
    local_chat_model_path = Path(r"D:\PYTHON\models\qwen2.5-1.5B-instruct")

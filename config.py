import os
from pathlib import Path
class Config:
    model_name = "deepseek-v4-flash"
    api_key = "sk-8f6f9069c25f48b8b4eb5325b1da253a"
    url = "https://api.deepseek.com/v1/"
    max_tokens = 2048
    temperature = 0.4
    embedding_model = "bpe"
    embedding_dim = 2560
    embedding_url = "https://api.deepseek.com/v1/embeddings"
    embedding_api_key = "sk-23333"
    data_path = Path("data/individual_simulation_data")

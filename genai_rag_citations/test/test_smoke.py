import os
import yaml
from pydantic import BaseModel




class Config(BaseModel):
    embedding: dict
    llm: dict
    retrieval: dict
    reranker: dict
    paths: dict

def test_config_example_loads():
    with open("configs/config.example.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    Config(**cfg)


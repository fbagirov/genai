import os, yaml
from functools import lru_cache

@lru_cache
def get_settings():
    cfg_path = os.getenv("CONFIG_PATH", "./configs/config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


import yaml
from pydantic import BaseModel

class Config(BaseModel):
    base_model_id: str
    tokenizer_id: str | None = None
    output_dir: str
    quantization: dict
    lora: dict
    tracking: dict
    sft: dict
    dpo: dict
    serve: dict | None = None

def test_config_example_loads():
    with open("configs/config.example.yaml","r") as f:
        cfg = yaml.safe_load(f)
    Config(**cfg)

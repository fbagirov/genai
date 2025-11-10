
from pathlib import Path
import yaml

def test_config_example_loads():
    cfg = yaml.safe_load(open("configs/config.example.yaml","r",encoding="utf-8"))
    assert "max_steps" in cfg
    assert "python_exec" in cfg

def test_layout():
    assert Path("run_agent.py").exists()
    assert Path("agent/planner.py").exists()
    assert Path("tools/python_exec.py").exists()

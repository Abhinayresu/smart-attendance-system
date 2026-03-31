import os
import yaml
from pathlib import Path

"""
Settings Module
---------------
Responsible for loading and providing application-level configurations.
Uses YAML for configuration storage and provides a centralized access point.
"""

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')

def load_config():
    """
    Loads configuration from the YAML file.
    Returns:
        dict: Configuration parameters.
    """
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

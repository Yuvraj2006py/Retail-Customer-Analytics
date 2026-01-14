"""
Utility helper functions for the retail analytics project.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration values
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, raises ValueError if not
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True

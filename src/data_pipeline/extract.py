"""
Data Extraction Module

This module handles loading data from various sources (CSV, Parquet)
with proper validation and error handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_transactions(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load transactional data from file.
    
    Args:
        file_path: Path to transactions CSV/Parquet file
        **kwargs: Additional arguments for pd.read_csv/read_parquet
        
    Returns:
        DataFrame with transaction data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Transactions file not found: {file_path}")
    
    logger.info(f"Loading transactions from {file_path}")
    
    # Determine file type and load accordingly
    if file_path.suffix.lower() == '.parquet':
        # Try different engines if default fails
        try:
            df = pd.read_parquet(file_path, engine='pyarrow', **kwargs)
        except Exception:
            try:
                df = pd.read_parquet(file_path, engine='fastparquet', **kwargs)
            except Exception:
                # Fallback to default
                df = pd.read_parquet(file_path, **kwargs)
    else:
        # Optimize CSV reading with appropriate dtypes
        df = pd.read_csv(
            file_path,
            parse_dates=['TransactionDate'],
            dtype={
                'TransactionID': 'string',
                'CustomerID': 'string',
                'StoreID': 'string',
                'ProductID': 'string',
                'Quantity': 'int32',
                'UnitPrice': 'float32',
                'LineTotal': 'float32'
            },
            **kwargs
        )
    
    # Validate required columns
    required_columns = [
        'TransactionID', 'CustomerID', 'TransactionDate',
        'StoreID', 'ProductID', 'Quantity', 'UnitPrice', 'LineTotal'
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Basic validation
    if len(df) == 0:
        raise ValueError("Transactions file is empty")
    
    logger.info(f"Loaded {len(df):,} transaction records")
    return df


def load_loyalty_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load PC Optimum loyalty program data.
    
    Args:
        file_path: Path to loyalty data CSV/Parquet file
        **kwargs: Additional arguments for pd.read_csv/read_parquet
        
    Returns:
        DataFrame with loyalty data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Loyalty file not found: {file_path}")
    
    logger.info(f"Loading loyalty data from {file_path}")
    
    if file_path.suffix.lower() == '.parquet':
        # Try different engines if default fails
        try:
            df = pd.read_parquet(file_path, engine='pyarrow', **kwargs)
        except Exception:
            try:
                df = pd.read_parquet(file_path, engine='fastparquet', **kwargs)
            except Exception:
                # Fallback to default
                df = pd.read_parquet(file_path, **kwargs)
    else:
        df = pd.read_csv(
            file_path,
            parse_dates=['TransactionDate'],
            dtype={
                'LoyaltyRecordID': 'int64',
                'CustomerID': 'string',
                'TransactionID': 'string',
                'PointsEarned': 'int32',
                'PointsRedeemed': 'int32',
                'PointsBalance': 'int64',
                'Tier': 'category',
                'PointsPerDollar': 'float32'
            },
            **kwargs
        )
    
    required_columns = [
        'LoyaltyRecordID', 'CustomerID', 'TransactionID',
        'TransactionDate', 'PointsEarned', 'PointsBalance', 'Tier'
    ]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if len(df) == 0:
        raise ValueError("Loyalty file is empty")
    
    logger.info(f"Loaded {len(df):,} loyalty records")
    return df


def load_survey_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load customer survey data.
    
    Args:
        file_path: Path to survey data CSV/Parquet file
        **kwargs: Additional arguments for pd.read_csv/read_parquet
        
    Returns:
        DataFrame with survey data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Survey file not found: {file_path}")
    
    logger.info(f"Loading survey data from {file_path}")
    
    if file_path.suffix.lower() == '.parquet':
        # Try different engines if default fails
        try:
            df = pd.read_parquet(file_path, engine='pyarrow', **kwargs)
        except Exception:
            try:
                df = pd.read_parquet(file_path, engine='fastparquet', **kwargs)
            except Exception:
                # Fallback to default
                df = pd.read_parquet(file_path, **kwargs)
    else:
        df = pd.read_csv(
            file_path,
            parse_dates=['SurveyDate'],
            dtype={
                'SurveyID': 'int64',
                'CustomerID': 'string',
                'SatisfactionScore': 'int8',
                'NPSScore': 'int8',
                'Feedback': 'string',
                'WouldRecommend': 'category'
            },
            **kwargs
        )
    
    required_columns = ['SurveyID', 'CustomerID', 'SurveyDate', 'SatisfactionScore', 'NPSScore']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Loaded {len(df):,} survey records")
    return df


def load_products(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load product catalog data.
    
    Args:
        file_path: Path to products CSV/Parquet file
        **kwargs: Additional arguments for pd.read_csv/read_parquet
        
    Returns:
        DataFrame with product data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Products file not found: {file_path}")
    
    logger.info(f"Loading products from {file_path}")
    
    if file_path.suffix.lower() == '.parquet':
        # Try different engines if default fails
        try:
            df = pd.read_parquet(file_path, engine='pyarrow', **kwargs)
        except Exception:
            try:
                df = pd.read_parquet(file_path, engine='fastparquet', **kwargs)
            except Exception:
                # Fallback to default
                df = pd.read_parquet(file_path, **kwargs)
    else:
        df = pd.read_csv(
            file_path,
            dtype={
                'ProductID': 'string',
                'ProductName': 'string',
                'Category': 'category',
                'Price': 'float32',
                'Cost': 'float32'
            },
            **kwargs
        )
    
    required_columns = ['ProductID', 'ProductName', 'Category', 'Price']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Loaded {len(df):,} products")
    return df


def load_stores(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load store location data.
    
    Args:
        file_path: Path to stores CSV/Parquet file
        **kwargs: Additional arguments for pd.read_csv/read_parquet
        
    Returns:
        DataFrame with store data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Stores file not found: {file_path}")
    
    logger.info(f"Loading stores from {file_path}")
    
    if file_path.suffix.lower() == '.parquet':
        # Try different engines if default fails
        try:
            df = pd.read_parquet(file_path, engine='pyarrow', **kwargs)
        except Exception:
            try:
                df = pd.read_parquet(file_path, engine='fastparquet', **kwargs)
            except Exception:
                # Fallback to default
                df = pd.read_parquet(file_path, **kwargs)
    else:
        df = pd.read_csv(
            file_path,
            dtype={
                'StoreID': 'string',
                'StoreName': 'string',
                'City': 'string',
                'Province': 'category',
                'Address': 'string'
            },
            **kwargs
        )
    
    required_columns = ['StoreID', 'StoreName', 'City', 'Province']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Loaded {len(df):,} stores")
    return df


def load_customers(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load customer data.
    
    Args:
        file_path: Path to customers CSV/Parquet file
        **kwargs: Additional arguments for pd.read_csv/read_parquet
        
    Returns:
        DataFrame with customer data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Customers file not found: {file_path}")
    
    logger.info(f"Loading customers from {file_path}")
    
    if file_path.suffix.lower() == '.parquet':
        # Try different engines if default fails
        try:
            df = pd.read_parquet(file_path, engine='pyarrow', **kwargs)
        except Exception:
            try:
                df = pd.read_parquet(file_path, engine='fastparquet', **kwargs)
            except Exception:
                # Fallback to default
                df = pd.read_parquet(file_path, **kwargs)
    else:
        df = pd.read_csv(
            file_path,
            parse_dates=['EnrollmentDate', 'DateOfBirth'],
            dtype={
                'CustomerID': 'string',
                'FirstName': 'string',
                'LastName': 'string',
                'Email': 'string',
                'Phone': 'string',
                'Address': 'string',
                'City': 'string',
                'Province': 'category',
                'PostalCode': 'string'
            },
            **kwargs
        )
    
    required_columns = ['CustomerID', 'FirstName', 'LastName', 'EnrollmentDate']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Loaded {len(df):,} customers")
    return df


def load_reference_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all reference data (products, stores, customers).
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with keys: 'products', 'stores', 'customers'
    """
    data_path = Path(data_dir)
    
    return {
        'products': load_products(data_path / 'products.csv'),
        'stores': load_stores(data_path / 'stores.csv'),
        'customers': load_customers(data_path / 'customers.csv')
    }


def load_all_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets from data directory.
    
    Args:
        data_dir: Directory containing all data files
        
    Returns:
        Dictionary with all loaded DataFrames
    """
    data_path = Path(data_dir)
    
    logger.info(f"Loading all data from {data_path}")
    
    datasets = {
        'transactions': load_transactions(data_path / 'transactions.csv'),
        'loyalty': load_loyalty_data(data_path / 'loyalty_pc_optimum.csv'),
        'surveys': load_survey_data(data_path / 'surveys.csv'),
        'products': load_products(data_path / 'products.csv'),
        'stores': load_stores(data_path / 'stores.csv'),
        'customers': load_customers(data_path / 'customers.csv')
    }
    
    logger.info("All data loaded successfully")
    return datasets


def validate_data_sources(data_dir: str) -> Dict[str, bool]:
    """
    Validate that all required data files exist.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with validation results for each file
    """
    data_path = Path(data_dir)
    required_files = [
        'transactions.csv',
        'loyalty_pc_optimum.csv',
        'surveys.csv',
        'products.csv',
        'stores.csv',
        'customers.csv'
    ]
    
    results = {}
    for file in required_files:
        file_path = data_path / file
        results[file] = file_path.exists()
        if not results[file]:
            logger.warning(f"Missing file: {file_path}")
    
    all_present = all(results.values())
    if all_present:
        logger.info("All required data files are present")
    else:
        missing = [f for f, exists in results.items() if not exists]
        raise FileNotFoundError(f"Missing required files: {missing}")
    
    return results

"""
Data Loading Module

This module handles saving processed data to files and databases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def save_processed_data(df: pd.DataFrame, 
                       file_path: str,
                       format: str = 'parquet',
                       **kwargs) -> None:
    """
    Save processed DataFrame to file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        format: File format ('parquet', 'csv', or 'both')
        **kwargs: Additional arguments for save functions
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(df):,} records to {file_path}")
    
    if format.lower() == 'parquet' or format.lower() == 'both':
        parquet_path = file_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, index=False, **kwargs)
        logger.info(f"Saved to {parquet_path}")
    
    if format.lower() == 'csv' or format.lower() == 'both':
        csv_path = file_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False, **kwargs)
        logger.info(f"Saved to {csv_path}")


def save_all_processed_data(datasets: Dict[str, pd.DataFrame],
                           output_dir: str,
                           format: str = 'parquet') -> None:
    """
    Save all processed datasets to files.
    
    Args:
        datasets: Dictionary of dataset names and DataFrames
        output_dir: Output directory
        format: File format ('parquet', 'csv', or 'both')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(datasets)} datasets to {output_path}")
    
    for name, df in datasets.items():
        file_path = output_path / name
        save_processed_data(df, file_path, format=format)
    
    logger.info("All datasets saved successfully")


def create_data_warehouse_structure(output_dir: str) -> None:
    """
    Create directory structure for data warehouse.
    
    Args:
        output_dir: Base output directory
    """
    output_path = Path(output_dir)
    
    # Create subdirectories
    (output_path / 'fact_tables').mkdir(parents=True, exist_ok=True)
    (output_path / 'dimension_tables').mkdir(parents=True, exist_ok=True)
    (output_path / 'aggregated_tables').mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Data warehouse structure created at {output_path}")


def prepare_fact_tables(transactions_df: pd.DataFrame,
                       loyalty_df: pd.DataFrame,
                       surveys_df: pd.DataFrame,
                       output_dir: str) -> None:
    """
    Prepare and save fact tables for data warehouse.
    
    Args:
        transactions_df: Transactions DataFrame
        loyalty_df: Loyalty DataFrame
        surveys_df: Surveys DataFrame
        output_dir: Output directory
    """
    output_path = Path(output_dir) / 'fact_tables'
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Preparing fact tables")
    
    # Fact Transactions
    fact_transactions = transactions_df[[
        'TransactionID', 'CustomerID', 'TransactionDate',
        'StoreID', 'ProductID', 'Quantity', 'UnitPrice', 'LineTotal'
    ]].copy()
    
    save_processed_data(fact_transactions, output_path / 'fact_transactions', format='parquet')
    
    # Fact Loyalty Points
    fact_loyalty = loyalty_df[[
        'LoyaltyRecordID', 'CustomerID', 'TransactionID',
        'TransactionDate', 'PointsEarned', 'PointsRedeemed',
        'PointsBalance', 'Tier', 'PointsPerDollar'
    ]].copy()
    
    save_processed_data(fact_loyalty, output_path / 'fact_loyalty_points', format='parquet')
    
    # Fact Surveys
    fact_surveys = surveys_df[[
        'SurveyID', 'CustomerID', 'SurveyDate',
        'SatisfactionScore', 'NPSScore', 'Feedback', 'WouldRecommend'
    ]].copy()
    
    save_processed_data(fact_surveys, output_path / 'fact_surveys', format='parquet')
    
    logger.info("Fact tables prepared and saved")


def prepare_dimension_tables(customers_df: pd.DataFrame,
                            products_df: pd.DataFrame,
                            stores_df: pd.DataFrame,
                            output_dir: str) -> None:
    """
    Prepare and save dimension tables for data warehouse.
    
    Args:
        customers_df: Customers DataFrame
        products_df: Products DataFrame
        stores_df: Stores DataFrame
        output_dir: Output directory
    """
    output_path = Path(output_dir) / 'dimension_tables'
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Preparing dimension tables")
    
    # Dim Customers
    dim_customers = customers_df.copy()
    save_processed_data(dim_customers, output_path / 'dim_customers', format='parquet')
    
    # Dim Products
    dim_products = products_df.copy()
    save_processed_data(dim_products, output_path / 'dim_products', format='parquet')
    
    # Dim Stores
    dim_stores = stores_df.copy()
    save_processed_data(dim_stores, output_path / 'dim_stores', format='parquet')
    
    # Dim Dates (create from transaction dates)
    # This would typically be created from all unique dates in the system
    logger.info("Dimension tables prepared and saved")


def create_date_dimension(start_date: str = '2020-01-01',
                          end_date: str = '2025-12-31',
                          output_dir: str = None) -> pd.DataFrame:
    """
    Create a date dimension table for time-based analysis.
    
    Args:
        start_date: Start date for dimension
        end_date: End date for dimension
        output_dir: Optional output directory to save
        
    Returns:
        Date dimension DataFrame
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    dim_dates = pd.DataFrame({
        'DateKey': date_range.strftime('%Y%m%d').astype(int),
        'Date': date_range,
        'Year': date_range.year,
        'Quarter': date_range.quarter,
        'Month': date_range.month,
        'MonthName': date_range.strftime('%B'),
        'Week': date_range.isocalendar().week,
        'DayOfWeek': date_range.dayofweek,
        'DayName': date_range.strftime('%A'),
        'DayOfMonth': date_range.day,
        'IsWeekend': date_range.dayofweek.isin([5, 6]),
        'IsHolidaySeason': date_range.month.isin([11, 12, 1, 2])  # Nov, Dec, Jan, Feb
    })
    
    if output_dir:
        output_path = Path(output_dir) / 'dimension_tables'
        output_path.mkdir(parents=True, exist_ok=True)
        save_processed_data(dim_dates, output_path / 'dim_dates', format='parquet')
        logger.info(f"Date dimension created and saved with {len(dim_dates):,} records")
    
    return dim_dates

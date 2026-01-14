"""
Data Transformation Module

This module handles data cleaning, validation, and transformation
with optimized operations to avoid bottlenecks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def clean_transactions(df: pd.DataFrame, 
                       remove_duplicates: bool = True,
                       handle_outliers: bool = True) -> pd.DataFrame:
    """
    Clean transactional data.
    
    Args:
        df: Transactions DataFrame
        remove_duplicates: Whether to remove duplicate transactions
        handle_outliers: Whether to handle price/quantity outliers
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning {len(df):,} transaction records")
    df = df.copy()
    
    initial_count = len(df)
    
    # 1. Remove rows with missing critical fields (OPTIMIZED: vectorized)
    critical_fields = ['TransactionID', 'CustomerID', 'ProductID', 'TransactionDate']
    missing_mask = df[critical_fields].isna().any(axis=1)
    if missing_mask.any():
        removed = missing_mask.sum()
        logger.warning(f"Removing {removed:,} transactions with missing critical fields")
        df = df[~missing_mask]
    
    # 2. Remove duplicate transactions (OPTIMIZED: use drop_duplicates)
    if remove_duplicates:
        duplicates = df.duplicated(subset=['TransactionID', 'CustomerID', 'ProductID', 'TransactionDate'], keep='first')
        if duplicates.any():
            removed = duplicates.sum()
            logger.warning(f"Removing {removed:,} duplicate transactions")
            df = df[~duplicates]
    
    # 3. Validate and fix data types
    # Ensure quantities are positive integers
    invalid_quantity = (df['Quantity'] <= 0) | (df['Quantity'] > 1000)
    if invalid_quantity.any():
        logger.warning(f"Fixing {invalid_quantity.sum():,} invalid quantities")
        df.loc[invalid_quantity, 'Quantity'] = 1
    
    # Ensure prices are positive
    invalid_price = df['UnitPrice'] <= 0
    if invalid_price.any():
        logger.warning(f"Removing {invalid_price.sum():,} transactions with invalid prices")
        df = df[~invalid_price]
    
    # 4. Handle outliers (OPTIMIZED: vectorized operations)
    if handle_outliers:
        # Price outliers (beyond 3 standard deviations)
        price_mean = df['UnitPrice'].mean()
        price_std = df['UnitPrice'].std()
        price_outliers = (df['UnitPrice'] > price_mean + 3 * price_std) | (df['UnitPrice'] < price_mean - 3 * price_std)
        if price_outliers.any():
            logger.warning(f"Flagging {price_outliers.sum():,} price outliers")
            # Cap outliers instead of removing
            upper_bound = price_mean + 3 * price_std
            lower_bound = max(0, price_mean - 3 * price_std)
            df.loc[price_outliers, 'UnitPrice'] = df.loc[price_outliers, 'UnitPrice'].clip(lower=lower_bound, upper=upper_bound)
            # Recalculate LineTotal
            df.loc[price_outliers, 'LineTotal'] = (df.loc[price_outliers, 'UnitPrice'] * 
                                                   df.loc[price_outliers, 'Quantity']).round(2)
    
    # 5. Validate LineTotal matches UnitPrice * Quantity (with small tolerance)
    calculated_total = (df['UnitPrice'] * df['Quantity']).round(2)
    mismatch = abs(df['LineTotal'] - calculated_total) > 0.01
    if mismatch.any():
        logger.warning(f"Fixing {mismatch.sum():,} LineTotal mismatches")
        df.loc[mismatch, 'LineTotal'] = calculated_total[mismatch]
    
    # 6. Validate transaction dates are within reasonable range
    min_date = pd.Timestamp('2020-01-01')
    max_date = pd.Timestamp('2025-12-31')
    invalid_dates = (df['TransactionDate'] < min_date) | (df['TransactionDate'] > max_date)
    if invalid_dates.any():
        logger.warning(f"Removing {invalid_dates.sum():,} transactions with invalid dates")
        df = df[~invalid_dates]
    
    final_count = len(df)
    removed = initial_count - final_count
    logger.info(f"Cleaning complete: {removed:,} records removed, {final_count:,} records remaining")
    
    return df.reset_index(drop=True)


def clean_loyalty_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean loyalty program data.
    
    Args:
        df: Loyalty DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning {len(df):,} loyalty records")
    df = df.copy()
    
    initial_count = len(df)
    
    # 1. Remove rows with missing critical fields
    critical_fields = ['CustomerID', 'TransactionID', 'TransactionDate']
    missing_mask = df[critical_fields].isna().any(axis=1)
    if missing_mask.any():
        removed = missing_mask.sum()
        logger.warning(f"Removing {removed:,} loyalty records with missing critical fields")
        df = df[~missing_mask]
    
    # 2. Validate points are non-negative
    invalid_points = (df['PointsEarned'] < 0) | (df['PointsRedeemed'] < 0) | (df['PointsBalance'] < 0)
    if invalid_points.any():
        logger.warning(f"Fixing {invalid_points.sum():,} negative point values")
        df.loc[invalid_points, 'PointsEarned'] = df.loc[invalid_points, 'PointsEarned'].clip(lower=0)
        df.loc[invalid_points, 'PointsRedeemed'] = df.loc[invalid_points, 'PointsRedeemed'].clip(lower=0)
        df.loc[invalid_points, 'PointsBalance'] = df.loc[invalid_points, 'PointsBalance'].clip(lower=0)
    
    # 3. Validate tier values
    valid_tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
    invalid_tier = ~df['Tier'].isin(valid_tiers)
    if invalid_tier.any():
        logger.warning(f"Fixing {invalid_tier.sum():,} invalid tier values")
        df.loc[invalid_tier, 'Tier'] = 'Bronze'  # Default to Bronze
    
    # 4. Flag suspicious redemptions (redeeming more than balance)
    suspicious = df['PointsRedeemed'] > df['PointsBalance'].shift(1).fillna(0)
    if suspicious.any():
        logger.warning(f"Flagging {suspicious.sum():,} suspicious redemptions")
        # This is expected in some cases due to transaction ordering, so we'll just log it
    
    final_count = len(df)
    removed = initial_count - final_count
    logger.info(f"Cleaning complete: {removed:,} records removed, {final_count:,} records remaining")
    
    return df.reset_index(drop=True)


def clean_survey_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean survey data.
    
    Args:
        df: Survey DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning {len(df):,} survey records")
    df = df.copy()
    
    initial_count = len(df)
    
    # 1. Remove rows with missing CustomerID
    missing_customer = df['CustomerID'].isna()
    if missing_customer.any():
        removed = missing_customer.sum()
        logger.warning(f"Removing {removed:,} surveys with missing CustomerID")
        df = df[~missing_customer]
    
    # 2. Validate satisfaction scores (1-5)
    invalid_satisfaction = (df['SatisfactionScore'] < 1) | (df['SatisfactionScore'] > 5)
    if invalid_satisfaction.any():
        logger.warning(f"Fixing {invalid_satisfaction.sum():,} invalid satisfaction scores")
        df.loc[invalid_satisfaction, 'SatisfactionScore'] = df.loc[invalid_satisfaction, 'SatisfactionScore'].clip(lower=1, upper=5)
    
    # 3. Validate NPS scores (0-10)
    invalid_nps = (df['NPSScore'] < 0) | (df['NPSScore'] > 10)
    if invalid_nps.any():
        logger.warning(f"Fixing {invalid_nps.sum():,} invalid NPS scores")
        df.loc[invalid_nps, 'NPSScore'] = df.loc[invalid_nps, 'NPSScore'].clip(lower=0, upper=10)
    
    # 4. Impute missing scores with median (if any remain)
    if df['SatisfactionScore'].isna().any():
        median_satisfaction = df['SatisfactionScore'].median()
        df['SatisfactionScore'].fillna(median_satisfaction, inplace=True)
        logger.info(f"Imputed missing satisfaction scores with median: {median_satisfaction}")
    
    if df['NPSScore'].isna().any():
        median_nps = df['NPSScore'].median()
        df['NPSScore'].fillna(median_nps, inplace=True)
        logger.info(f"Imputed missing NPS scores with median: {median_nps}")
    
    final_count = len(df)
    removed = initial_count - final_count
    logger.info(f"Cleaning complete: {removed:,} records removed, {final_count:,} records remaining")
    
    return df.reset_index(drop=True)


def normalize_dates(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """
    Normalize date columns to datetime format.
    
    Args:
        df: DataFrame with date columns
        date_columns: List of column names to normalize
        
    Returns:
        DataFrame with normalized dates
    """
    df = df.copy()
    
    for col in date_columns:
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
                invalid_dates = df[col].isna()
                if invalid_dates.any():
                    logger.warning(f"Found {invalid_dates.sum():,} invalid dates in {col}")
    
    return df


def validate_referential_integrity(transactions_df: pd.DataFrame,
                                   loyalty_df: pd.DataFrame,
                                   surveys_df: pd.DataFrame,
                                   customers_df: pd.DataFrame,
                                   products_df: pd.DataFrame,
                                   stores_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate referential integrity between datasets.
    
    Args:
        transactions_df: Transactions DataFrame
        loyalty_df: Loyalty DataFrame
        surveys_df: Surveys DataFrame
        customers_df: Customers DataFrame
        products_df: Products DataFrame
        stores_df: Stores DataFrame
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating referential integrity")
    results = {}
    
    # Get unique IDs
    customer_ids = set(customers_df['CustomerID'].unique())
    product_ids = set(products_df['ProductID'].unique())
    store_ids = set(stores_df['StoreID'].unique())
    transaction_ids = set(transactions_df['TransactionID'].unique())
    
    # Validate transactions reference valid customers
    trans_customers = set(transactions_df['CustomerID'].unique())
    invalid_customers = trans_customers - customer_ids
    results['transactions_customers'] = len(invalid_customers) == 0
    if invalid_customers:
        logger.warning(f"Found {len(invalid_customers):,} invalid customer references in transactions")
    
    # Validate transactions reference valid products
    trans_products = set(transactions_df['ProductID'].unique())
    invalid_products = trans_products - product_ids
    results['transactions_products'] = len(invalid_products) == 0
    if invalid_products:
        logger.warning(f"Found {len(invalid_products):,} invalid product references in transactions")
    
    # Validate transactions reference valid stores
    trans_stores = set(transactions_df['StoreID'].unique())
    invalid_stores = trans_stores - store_ids
    results['transactions_stores'] = len(invalid_stores) == 0
    if invalid_stores:
        logger.warning(f"Found {len(invalid_stores):,} invalid store references in transactions")
    
    # Validate loyalty records reference valid transactions
    loyalty_trans = set(loyalty_df['TransactionID'].unique())
    invalid_loyalty_trans = loyalty_trans - transaction_ids
    results['loyalty_transactions'] = len(invalid_loyalty_trans) == 0
    if invalid_loyalty_trans:
        logger.warning(f"Found {len(invalid_loyalty_trans):,} invalid transaction references in loyalty data")
    
    # Validate loyalty records reference valid customers
    loyalty_customers = set(loyalty_df['CustomerID'].unique())
    invalid_loyalty_customers = loyalty_customers - customer_ids
    results['loyalty_customers'] = len(invalid_loyalty_customers) == 0
    if invalid_loyalty_customers:
        logger.warning(f"Found {len(invalid_loyalty_customers):,} invalid customer references in loyalty data")
    
    # Validate surveys reference valid customers
    survey_customers = set(surveys_df['CustomerID'].unique())
    invalid_survey_customers = survey_customers - customer_ids
    results['surveys_customers'] = len(invalid_survey_customers) == 0
    if invalid_survey_customers:
        logger.warning(f"Found {len(invalid_survey_customers):,} invalid customer references in surveys")
    
    all_valid = all(results.values())
    if all_valid:
        logger.info("All referential integrity checks passed")
    else:
        logger.warning("Some referential integrity checks failed")
    
    return results


def merge_datasets(transactions_df: pd.DataFrame,
                   loyalty_df: pd.DataFrame,
                   surveys_df: pd.DataFrame,
                   customers_df: pd.DataFrame,
                   products_df: pd.DataFrame,
                   stores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all datasets into a single analytical dataset.
    
    Args:
        transactions_df: Cleaned transactions DataFrame
        loyalty_df: Cleaned loyalty DataFrame
        surveys_df: Cleaned surveys DataFrame
        customers_df: Customers DataFrame
        products_df: Products DataFrame
        stores_df: Stores DataFrame
        
    Returns:
        Merged DataFrame
    """
    logger.info("Merging datasets")
    
    # Start with transactions as base
    merged = transactions_df.copy()
    
    # Merge with products (OPTIMIZED: use merge instead of repeated lookups)
    merged = merged.merge(
        products_df[['ProductID', 'ProductName', 'Category', 'Price', 'Cost']],
        on='ProductID',
        how='left',
        suffixes=('', '_product')
    )
    
    # Merge with stores
    merged = merged.merge(
        stores_df[['StoreID', 'StoreName', 'City', 'Province']],
        on='StoreID',
        how='left'
    )
    
    # Merge with customers
    merged = merged.merge(
        customers_df[['CustomerID', 'FirstName', 'LastName', 'EnrollmentDate', 'City', 'Province']],
        on='CustomerID',
        how='left',
        suffixes=('', '_customer')
    )
    
    # Aggregate loyalty data per transaction (some transactions may have multiple loyalty records)
    loyalty_agg = loyalty_df.groupby('TransactionID').agg({
        'PointsEarned': 'sum',
        'PointsRedeemed': 'sum',
        'PointsBalance': 'last',  # Take the last balance for the transaction
        'Tier': 'last',
        'PointsPerDollar': 'last'
    }).reset_index()
    
    merged = merged.merge(
        loyalty_agg,
        on='TransactionID',
        how='left'
    )
    
    # For surveys, we'll create a customer-level summary (since surveys are per customer, not transaction)
    survey_summary = surveys_df.groupby('CustomerID').agg({
        'SatisfactionScore': 'mean',
        'NPSScore': 'mean',
        'WouldRecommend': lambda x: 'Yes' if (x == 'Yes').any() else 'No'
    }).reset_index()
    
    merged = merged.merge(
        survey_summary,
        on='CustomerID',
        how='left'
    )
    
    logger.info(f"Merged dataset created with {len(merged):,} records")
    return merged


def calculate_data_quality_metrics(df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, float]:
    """
    Calculate data quality metrics for a dataset.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset for logging
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isna().sum().sum(),
        'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_records': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
    }
    
    logger.info(f"Data quality metrics for {dataset_name}:")
    logger.info(f"  Total records: {metrics['total_records']:,}")
    logger.info(f"  Missing values: {metrics['missing_values']:,} ({metrics['missing_percentage']:.2f}%)")
    logger.info(f"  Duplicate records: {metrics['duplicate_records']:,} ({metrics['duplicate_percentage']:.2f}%)")
    
    return metrics

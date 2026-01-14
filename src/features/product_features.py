"""
Product Feature Engineering Module

This module calculates product-level features including:
- Product performance metrics
- Category analysis
- Customer-product affinity
- Cross-sell opportunities
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_product_performance(transactions_df: pd.DataFrame,
                                 products_df: pd.DataFrame,
                                 reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Calculate product performance metrics.
    
    Args:
        transactions_df: DataFrame with transaction data
        products_df: DataFrame with product information
        reference_date: Reference date for calculations
        
    Returns:
        DataFrame with product performance features
    """
    if reference_date is None:
        reference_date = transactions_df['TransactionDate'].max()
    
    logger.info("Calculating product performance metrics")
    
    # Merge transactions with products
    trans_with_products = transactions_df.merge(
        products_df[['ProductID', 'ProductName', 'Category', 'Price', 'Cost']],
        on='ProductID',
        how='left'
    )
    
    # Aggregate product metrics
    product_perf = trans_with_products.groupby('ProductID').agg({
        'TransactionID': 'count',  # Number of transactions
        'Quantity': ['sum', 'mean'],  # Total units sold, average quantity
        'LineTotal': ['sum', 'mean'],  # Total revenue, average revenue per transaction
        'TransactionDate': ['min', 'max']  # First sale, last sale
    }).reset_index()
    
    # Flatten column names
    product_perf.columns = ['ProductID', 'TransactionCount', 'TotalUnitsSold',
                           'AvgQuantityPerTransaction', 'TotalRevenue', 'AvgRevenuePerTransaction',
                           'FirstSaleDate', 'LastSaleDate']
    
    # Merge with product details
    product_perf = product_perf.merge(
        products_df[['ProductID', 'ProductName', 'Category', 'Price', 'Cost']],
        on='ProductID',
        how='left'
    )
    
    # Calculate additional metrics
    product_perf['RevenuePerUnit'] = (
        product_perf['TotalRevenue'] / product_perf['TotalUnitsSold']
    ).replace([np.inf, -np.inf], np.nan).fillna(product_perf['Price'])
    
    product_perf['ProfitMargin'] = (
        (product_perf['Price'] - product_perf['Cost']) / product_perf['Price']
    ).fillna(0)
    
    # TotalProfit should use actual revenue, not price * units
    # Revenue already accounts for actual transaction prices
    product_perf['TotalProfit'] = (
        product_perf['TotalRevenue'] - (product_perf['Cost'] * product_perf['TotalUnitsSold'])
    )
    
    # Calculate product age (days since first sale)
    product_perf['ProductAge'] = (
        reference_date - product_perf['FirstSaleDate']
    ).dt.days
    
    # Calculate sales velocity (units per day)
    product_perf['SalesVelocity'] = (
        product_perf['TotalUnitsSold'] / product_perf['ProductAge'].clip(lower=1)
    )
    
    logger.info(f"Product performance calculated for {len(product_perf)} products")
    
    return product_perf


def calculate_category_performance(transactions_df: pd.DataFrame,
                                  products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate category-level performance metrics.
    
    Args:
        transactions_df: DataFrame with transaction data
        products_df: DataFrame with product information
        
    Returns:
        DataFrame with category performance features
    """
    logger.info("Calculating category performance metrics")
    
    # Merge transactions with products
    trans_with_products = transactions_df.merge(
        products_df[['ProductID', 'Category']],
        on='ProductID',
        how='left'
    )
    
    # Aggregate by category
    category_perf = trans_with_products.groupby('Category').agg({
        'TransactionID': 'count',
        'CustomerID': 'nunique',
        'ProductID': 'nunique',
        'Quantity': 'sum',
        'LineTotal': ['sum', 'mean']
    }).reset_index()
    
    # Flatten column names
    category_perf.columns = ['Category', 'TransactionCount', 'UniqueCustomers',
                           'UniqueProducts', 'TotalUnitsSold', 'TotalRevenue', 'AvgTransactionValue']
    
    # Calculate additional metrics
    category_perf['RevenuePerCustomer'] = (
        category_perf['TotalRevenue'] / category_perf['UniqueCustomers']
    )
    
    category_perf['RevenuePerProduct'] = (
        category_perf['TotalRevenue'] / category_perf['UniqueProducts']
    )
    
    # Calculate category share of total revenue
    total_revenue = category_perf['TotalRevenue'].sum()
    category_perf['RevenueShare'] = (
        category_perf['TotalRevenue'] / total_revenue
    )
    
    logger.info(f"Category performance calculated for {len(category_perf)} categories")
    
    return category_perf.sort_values('TotalRevenue', ascending=False)


def calculate_product_trends(transactions_df: pd.DataFrame,
                            products_df: pd.DataFrame,
                            reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Calculate product sales trends (growth/decline).
    
    Args:
        transactions_df: DataFrame with transaction data
        products_df: DataFrame with product information
        reference_date: Reference date for trend calculations
        
    Returns:
        DataFrame with product trend features
    """
    if reference_date is None:
        reference_date = transactions_df['TransactionDate'].max()
    
    logger.info("Calculating product trends")
    
    # Define periods
    three_months_ago = reference_date - timedelta(days=90)
    six_months_ago = reference_date - timedelta(days=180)
    
    # Recent period (last 3 months)
    recent_trans = transactions_df[transactions_df['TransactionDate'] >= three_months_ago]
    recent_sales = recent_trans.groupby('ProductID').agg({
        'Quantity': 'sum',
        'LineTotal': 'sum'
    }).reset_index()
    recent_sales.columns = ['ProductID', 'RecentUnitsSold', 'RecentRevenue']
    
    # Historical period (3-6 months ago)
    historical_trans = transactions_df[
        (transactions_df['TransactionDate'] >= six_months_ago) &
        (transactions_df['TransactionDate'] < three_months_ago)
    ]
    historical_sales = historical_trans.groupby('ProductID').agg({
        'Quantity': 'sum',
        'LineTotal': 'sum'
    }).reset_index()
    historical_sales.columns = ['ProductID', 'HistoricalUnitsSold', 'HistoricalRevenue']
    
    # Merge and calculate trends
    trends = recent_sales.merge(historical_sales, on='ProductID', how='outer').fillna(0)
    
    # Calculate growth rates
    trends['UnitsGrowthRate'] = (
        (trends['RecentUnitsSold'] - trends['HistoricalUnitsSold']) /
        (trends['HistoricalUnitsSold'] + 1)  # Add 1 to avoid division by zero
    )
    
    trends['RevenueGrowthRate'] = (
        (trends['RecentRevenue'] - trends['HistoricalRevenue']) /
        (trends['HistoricalRevenue'] + 1)
    )
    
    # Categorize trends
    trends['TrendDirection'] = trends['RevenueGrowthRate'].apply(
        lambda x: 'growing' if x > 0.1 else ('declining' if x < -0.1 else 'stable')
    )
    
    logger.info(f"Product trends calculated for {len(trends)} products")
    
    return trends[['ProductID', 'RecentUnitsSold', 'RecentRevenue',
                   'HistoricalUnitsSold', 'HistoricalRevenue',
                   'UnitsGrowthRate', 'RevenueGrowthRate', 'TrendDirection']]


def calculate_customer_product_affinity(transactions_df: pd.DataFrame,
                                       products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate customer-product affinity scores.
    
    Args:
        transactions_df: DataFrame with transaction data
        products_df: DataFrame with product information
        
    Returns:
        DataFrame with customer-product affinity metrics
    """
    logger.info("Calculating customer-product affinity")
    
    # Calculate purchase frequency per customer-product pair
    affinity = transactions_df.groupby(['CustomerID', 'ProductID']).agg({
        'TransactionID': 'count',
        'Quantity': 'sum',
        'LineTotal': 'sum',
        'TransactionDate': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    affinity.columns = ['CustomerID', 'ProductID', 'PurchaseCount', 'TotalQuantity',
                       'TotalSpend', 'FirstPurchaseDate', 'LastPurchaseDate']
    
    # Calculate days since last purchase
    reference_date = transactions_df['TransactionDate'].max()
    affinity['DaysSinceLastPurchase'] = (
        reference_date - affinity['LastPurchaseDate']
    ).dt.days
    
    # Calculate purchase recency score (inverse of days, normalized)
    max_days = affinity['DaysSinceLastPurchase'].max()
    affinity['RecencyScore'] = 1 - (affinity['DaysSinceLastPurchase'] / (max_days + 1))
    
    # Calculate frequency score (normalized)
    max_frequency = affinity['PurchaseCount'].max()
    affinity['FrequencyScore'] = affinity['PurchaseCount'] / (max_frequency + 1)
    
    # Calculate monetary score (normalized)
    max_spend = affinity['TotalSpend'].max()
    affinity['MonetaryScore'] = affinity['TotalSpend'] / (max_spend + 1)
    
    # Combined affinity score
    affinity['AffinityScore'] = (
        affinity['RecencyScore'] * 0.3 +
        affinity['FrequencyScore'] * 0.4 +
        affinity['MonetaryScore'] * 0.3
    )
    
    logger.info(f"Customer-product affinity calculated for {len(affinity)} customer-product pairs")
    
    return affinity


def calculate_cross_sell_opportunities(transactions_df: pd.DataFrame,
                                      products_df: pd.DataFrame,
                                      min_cooccurrence: int = 10) -> pd.DataFrame:
    """
    Identify cross-sell opportunities (products frequently bought together).
    
    Args:
        transactions_df: DataFrame with transaction data
        products_df: DataFrame with product information
        min_cooccurrence: Minimum number of co-occurrences to consider
        
    Returns:
        DataFrame with cross-sell opportunities
    """
    logger.info("Calculating cross-sell opportunities")
    
    # Get products per transaction
    transaction_products = transactions_df.groupby('TransactionID')['ProductID'].apply(
        lambda x: list(x.unique())
    ).reset_index()
    
    # Find product pairs that appear together
    product_pairs = []
    for products in transaction_products['ProductID']:
        if len(products) > 1:
            # Generate all pairs
            for i in range(len(products)):
                for j in range(i + 1, len(products)):
                    product_pairs.append(tuple(sorted([products[i], products[j]])))
    
    # Count co-occurrences
    if product_pairs:
        pairs_df = pd.DataFrame(product_pairs, columns=['Product1', 'Product2'])
        cooccurrence = pairs_df.groupby(['Product1', 'Product2']).size().reset_index(name='CoOccurrence')
        
        # Filter by minimum co-occurrence
        cooccurrence = cooccurrence[cooccurrence['CoOccurrence'] >= min_cooccurrence]
        
        # Add product names
        product_names = products_df.set_index('ProductID')['ProductName'].to_dict()
        cooccurrence['Product1Name'] = cooccurrence['Product1'].map(product_names)
        cooccurrence['Product2Name'] = cooccurrence['Product2'].map(product_names)
        
        cooccurrence = cooccurrence.sort_values('CoOccurrence', ascending=False)
        
        logger.info(f"Found {len(cooccurrence)} cross-sell opportunities")
    else:
        cooccurrence = pd.DataFrame(columns=['Product1', 'Product2', 'CoOccurrence',
                                           'Product1Name', 'Product2Name'])
        logger.warning("No cross-sell opportunities found")
    
    return cooccurrence


def create_product_feature_matrix(transactions_df: pd.DataFrame,
                                  products_df: pd.DataFrame,
                                  reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Create comprehensive product feature matrix.
    
    Args:
        transactions_df: Transaction data
        products_df: Product data
        reference_date: Reference date for calculations
        
    Returns:
        Complete product feature matrix
    """
    if reference_date is None:
        reference_date = transactions_df['TransactionDate'].max()
    
    logger.info("Creating comprehensive product feature matrix")
    
    # Start with product base
    feature_matrix = products_df[['ProductID', 'ProductName', 'Category', 'Price', 'Cost']].copy()
    
    # Calculate all feature sets
    logger.info("  - Calculating product performance...")
    perf_features = calculate_product_performance(transactions_df, products_df, reference_date)
    feature_matrix = feature_matrix.merge(
        perf_features.drop(columns=['ProductName', 'Category', 'Price', 'Cost'], errors='ignore'),
        on='ProductID',
        how='left'
    )
    
    logger.info("  - Calculating product trends...")
    trend_features = calculate_product_trends(transactions_df, products_df, reference_date)
    feature_matrix = feature_matrix.merge(trend_features, on='ProductID', how='left')
    
    # Fill missing values
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    feature_matrix[numeric_cols] = feature_matrix[numeric_cols].fillna(0)
    
    logger.info(f"Product feature matrix created with {len(feature_matrix)} products and {len(feature_matrix.columns)} features")
    
    return feature_matrix

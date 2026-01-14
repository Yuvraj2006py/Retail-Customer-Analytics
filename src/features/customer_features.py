"""
Customer Feature Engineering Module

This module calculates comprehensive customer-level features including:
- RFM (Recency, Frequency, Monetary) features
- Loyalty program features
- Engagement features
- Behavioral features
- Seasonal features
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_rfm_features(transactions_df: pd.DataFrame,
                           reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) features for each customer.
    
    Args:
        transactions_df: DataFrame with transaction data
        reference_date: Reference date for recency calculation (default: max transaction date)
        
    Returns:
        DataFrame with CustomerID and RFM features
    """
    if reference_date is None:
        reference_date = transactions_df['TransactionDate'].max()
    
    logger.info(f"Calculating RFM features with reference date: {reference_date}")
    
    # Group by customer and calculate metrics
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TransactionDate': ['max', 'min', 'count'],  # Last purchase, first purchase, frequency
        'LineTotal': ['sum', 'mean']  # Total monetary, average transaction value
    }).reset_index()
    
    # Flatten column names
    customer_metrics.columns = ['CustomerID', 'LastPurchaseDate', 'FirstPurchaseDate', 
                                'Frequency', 'TotalSpend', 'AvgTransactionValue']
    
    # Calculate Recency (days since last purchase)
    customer_metrics['Recency'] = (reference_date - customer_metrics['LastPurchaseDate']).dt.days
    
    # Calculate customer lifetime (days between first and last purchase)
    customer_metrics['CustomerLifetime'] = (
        customer_metrics['LastPurchaseDate'] - customer_metrics['FirstPurchaseDate']
    ).dt.days
    
    # Calculate additional monetary metrics
    customer_metrics['LifetimeValue'] = customer_metrics['TotalSpend']
    customer_metrics['AvgDaysBetweenPurchases'] = (
        customer_metrics['CustomerLifetime'] / customer_metrics['Frequency']
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate RFM scores (1-5 scale using quintiles)
    customer_metrics['RecencyScore'] = pd.qcut(
        customer_metrics['Recency'].rank(method='first'),
        q=5, labels=[5, 4, 3, 2, 1], duplicates='drop'
    ).astype(int)
    
    customer_metrics['FrequencyScore'] = pd.qcut(
        customer_metrics['Frequency'].rank(method='first'),
        q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
    ).astype(int)
    
    customer_metrics['MonetaryScore'] = pd.qcut(
        customer_metrics['TotalSpend'].rank(method='first'),
        q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
    ).astype(int)
    
    # Combined RFM score
    customer_metrics['RFMScore'] = (
        customer_metrics['RecencyScore'] * 100 +
        customer_metrics['FrequencyScore'] * 10 +
        customer_metrics['MonetaryScore']
    )
    
    logger.info(f"RFM features calculated for {len(customer_metrics)} customers")
    
    return customer_metrics[['CustomerID', 'Recency', 'Frequency', 'TotalSpend',
                             'AvgTransactionValue', 'LifetimeValue', 'CustomerLifetime',
                             'AvgDaysBetweenPurchases', 'RecencyScore', 'FrequencyScore',
                             'MonetaryScore', 'RFMScore', 'LastPurchaseDate', 'FirstPurchaseDate']]


def calculate_frequency_by_period(transactions_df: pd.DataFrame,
                                   reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Calculate transaction frequency for different time periods.
    
    Args:
        transactions_df: DataFrame with transaction data
        reference_date: Reference date for period calculations
        
    Returns:
        DataFrame with frequency metrics for 30, 60, 90, 365 days
    """
    if reference_date is None:
        reference_date = transactions_df['TransactionDate'].max()
    
    periods = [30, 60, 90, 365]
    period_data = []
    
    for period in periods:
        cutoff_date = reference_date - timedelta(days=period)
        period_transactions = transactions_df[
            transactions_df['TransactionDate'] >= cutoff_date
        ]
        
        period_freq = period_transactions.groupby('CustomerID').agg({
            'TransactionID': 'count',
            'LineTotal': 'sum'
        }).reset_index()
        
        period_freq.columns = ['CustomerID', f'Frequency_{period}d', f'Spend_{period}d']
        
        if len(period_data) == 0:
            period_data.append(period_freq)
        else:
            period_data[0] = period_data[0].merge(period_freq, on='CustomerID', how='outer')
    
    result = period_data[0].fillna(0)
    logger.info(f"Frequency by period calculated for {len(result)} customers")
    
    return result


def calculate_loyalty_features(loyalty_df: pd.DataFrame,
                               customers_df: pd.DataFrame,
                               reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Calculate loyalty program features for each customer.
    
    Args:
        loyalty_df: DataFrame with loyalty data
        customers_df: DataFrame with customer enrollment dates
        reference_date: Reference date for calculations
        
    Returns:
        DataFrame with loyalty features
    """
    if reference_date is None:
        reference_date = loyalty_df['TransactionDate'].max()
    
    logger.info("Calculating loyalty features")
    
    # Get latest loyalty record per customer
    latest_loyalty = loyalty_df.sort_values('TransactionDate').groupby('CustomerID').last().reset_index()
    
    # Aggregate loyalty metrics per customer
    loyalty_agg = loyalty_df.groupby('CustomerID').agg({
        'PointsEarned': ['sum', 'mean', 'count'],
        'PointsRedeemed': ['sum', 'mean'],
        'PointsBalance': 'last',
        'Tier': 'last',
        'PointsPerDollar': 'last'
    }).reset_index()
    
    # Flatten column names
    loyalty_agg.columns = ['CustomerID', 'TotalPointsEarned', 'AvgPointsEarned', 
                          'PointsTransactions', 'TotalPointsRedeemed', 'AvgPointsRedeemed',
                          'CurrentPointsBalance', 'CurrentTier', 'CurrentPointsPerDollar']
    
    # Calculate redemption rate
    loyalty_agg['RedemptionRate'] = (
        loyalty_agg['TotalPointsRedeemed'] / 
        loyalty_agg['TotalPointsEarned'].replace(0, np.nan)
    ).fillna(0)
    
    # Calculate days since enrollment
    if 'EnrollmentDate' in customers_df.columns:
        customers_enrollment = customers_df[['CustomerID', 'EnrollmentDate']].copy()
        customers_enrollment['DaysSinceEnrollment'] = (
            reference_date - pd.to_datetime(customers_enrollment['EnrollmentDate'])
        ).dt.days
        
        loyalty_agg = loyalty_agg.merge(customers_enrollment, on='CustomerID', how='left')
    
    # Calculate points per dollar ratio
    loyalty_agg['PointsPerDollarRatio'] = loyalty_agg['CurrentPointsPerDollar']
    
    # Tier encoding (numeric for ML)
    tier_mapping = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
    loyalty_agg['TierNumeric'] = loyalty_agg['CurrentTier'].map(tier_mapping).fillna(1)
    
    # Calculate tier history (number of tier changes)
    tier_changes = loyalty_df.groupby('CustomerID')['Tier'].apply(
        lambda x: (x != x.shift()).sum() - 1  # Subtract 1 because first tier isn't a change
    ).reset_index()
    tier_changes.columns = ['CustomerID', 'TierChanges']
    
    loyalty_agg = loyalty_agg.merge(tier_changes, on='CustomerID', how='left')
    loyalty_agg['TierChanges'] = loyalty_agg['TierChanges'].fillna(0)
    
    logger.info(f"Loyalty features calculated for {len(loyalty_agg)} customers")
    
    return loyalty_agg


def calculate_engagement_features(transactions_df: pd.DataFrame,
                                  products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate customer engagement features.
    
    Args:
        transactions_df: DataFrame with transaction data
        products_df: DataFrame with product information
        
    Returns:
        DataFrame with engagement features
    """
    logger.info("Calculating engagement features")
    
    # Merge with products to get categories
    trans_with_products = transactions_df.merge(
        products_df[['ProductID', 'Category']],
        on='ProductID',
        how='left'
    )
    
    # Group by customer
    engagement = trans_with_products.groupby('CustomerID').agg({
        'TransactionDate': ['nunique', 'min', 'max'],  # Days active, first, last
        'ProductID': 'nunique',  # Unique products
        'Category': 'nunique',  # Unique categories
        'Quantity': 'sum',  # Total items purchased
        'TransactionID': 'count'  # Total transactions
    }).reset_index()
    
    # Flatten column names
    engagement.columns = ['CustomerID', 'DaysActive', 'FirstTransactionDate', 
                          'LastTransactionDate', 'UniqueProducts', 'UniqueCategories',
                          'TotalItemsPurchased', 'TotalTransactions']
    
    # Calculate active months
    engagement['ActiveMonths'] = (
        (engagement['LastTransactionDate'] - engagement['FirstTransactionDate']).dt.days / 30.44
    ).round().astype(int).clip(lower=1)
    
    # Calculate average basket size
    basket_sizes = trans_with_products.groupby(['CustomerID', 'TransactionID'])['Quantity'].sum().reset_index()
    engagement = engagement.merge(
        basket_sizes.groupby('CustomerID')['Quantity'].mean().reset_index(),
        on='CustomerID',
        how='left'
    )
    engagement.rename(columns={'Quantity': 'AvgBasketSize'}, inplace=True)
    
    # Calculate purchase velocity (transactions per month)
    engagement['PurchaseVelocity'] = (
        engagement['TotalTransactions'] / engagement['ActiveMonths']
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate product diversity (unique products per transaction)
    engagement['ProductDiversity'] = (
        engagement['UniqueProducts'] / engagement['TotalTransactions']
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    logger.info(f"Engagement features calculated for {len(engagement)} customers")
    
    return engagement[['CustomerID', 'DaysActive', 'ActiveMonths', 'UniqueProducts',
                      'UniqueCategories', 'TotalItemsPurchased', 'TotalTransactions',
                      'AvgBasketSize', 'PurchaseVelocity', 'ProductDiversity']]


def calculate_behavioral_features(transactions_df: pd.DataFrame,
                                 reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Calculate behavioral features including trends.
    
    Args:
        transactions_df: DataFrame with transaction data
        reference_date: Reference date for trend calculations
        
    Returns:
        DataFrame with behavioral features
    """
    if reference_date is None:
        reference_date = transactions_df['TransactionDate'].max()
    
    logger.info("Calculating behavioral features")
    
    # Split into recent and historical periods for trend analysis
    six_months_ago = reference_date - timedelta(days=180)
    three_months_ago = reference_date - timedelta(days=90)
    
    # Recent period (last 3 months)
    recent_trans = transactions_df[transactions_df['TransactionDate'] >= three_months_ago]
    recent_metrics = recent_trans.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'LineTotal': ['sum', 'mean'],
        'Quantity': 'mean'
    }).reset_index()
    recent_metrics.columns = ['CustomerID', 'RecentFrequency', 'RecentSpend', 
                             'RecentAvgTransaction', 'RecentAvgBasketSize']
    
    # Historical period (3-6 months ago)
    historical_trans = transactions_df[
        (transactions_df['TransactionDate'] >= six_months_ago) &
        (transactions_df['TransactionDate'] < three_months_ago)
    ]
    historical_metrics = historical_trans.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'LineTotal': ['sum', 'mean'],
        'Quantity': 'mean'
    }).reset_index()
    historical_metrics.columns = ['CustomerID', 'HistoricalFrequency', 'HistoricalSpend',
                                 'HistoricalAvgTransaction', 'HistoricalAvgBasketSize']
    
    # Merge and calculate trends
    behavioral = recent_metrics.merge(historical_metrics, on='CustomerID', how='outer').fillna(0)
    
    # Calculate trend indicators
    behavioral['FrequencyTrend'] = (
        (behavioral['RecentFrequency'] - behavioral['HistoricalFrequency']) /
        (behavioral['HistoricalFrequency'] + 1)  # Add 1 to avoid division by zero
    )
    
    behavioral['SpendingTrend'] = (
        (behavioral['RecentSpend'] - behavioral['HistoricalSpend']) /
        (behavioral['HistoricalSpend'] + 1)
    )
    
    behavioral['BasketSizeTrend'] = (
        (behavioral['RecentAvgBasketSize'] - behavioral['HistoricalAvgBasketSize']) /
        (behavioral['HistoricalAvgBasketSize'] + 1)
    )
    
    # Categorize trends
    behavioral['FrequencyTrendDirection'] = behavioral['FrequencyTrend'].apply(
        lambda x: 'increasing' if x > 0.1 else ('decreasing' if x < -0.1 else 'stable')
    )
    
    behavioral['SpendingTrendDirection'] = behavioral['SpendingTrend'].apply(
        lambda x: 'increasing' if x > 0.1 else ('decreasing' if x < -0.1 else 'stable')
    )
    
    logger.info(f"Behavioral features calculated for {len(behavioral)} customers")
    
    return behavioral[['CustomerID', 'RecentFrequency', 'RecentSpend', 'RecentAvgTransaction',
                      'RecentAvgBasketSize', 'HistoricalFrequency', 'HistoricalSpend',
                      'HistoricalAvgTransaction', 'HistoricalAvgBasketSize',
                      'FrequencyTrend', 'SpendingTrend', 'BasketSizeTrend',
                      'FrequencyTrendDirection', 'SpendingTrendDirection']]


def calculate_seasonal_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate seasonal purchase patterns.
    
    Args:
        transactions_df: DataFrame with transaction data
        
    Returns:
        DataFrame with seasonal features
    """
    logger.info("Calculating seasonal features")
    
    # Extract month and quarter from transaction dates
    transactions_df = transactions_df.copy()
    transactions_df['Month'] = transactions_df['TransactionDate'].dt.month
    transactions_df['Quarter'] = transactions_df['TransactionDate'].dt.quarter
    transactions_df['IsHolidaySeason'] = transactions_df['Month'].isin([11, 12, 1, 2])
    
    # Group by customer and calculate seasonal metrics
    seasonal = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'LineTotal': 'sum',
        'IsHolidaySeason': 'sum'
    }).reset_index()
    
    seasonal.columns = ['CustomerID', 'TotalTransactions', 'TotalSpend', 'HolidayTransactions']
    
    # Calculate holiday season activity ratio
    seasonal['HolidayActivityRatio'] = (
        seasonal['HolidayTransactions'] / seasonal['TotalTransactions']
    ).fillna(0)
    
    # Calculate transactions by quarter
    quarterly = transactions_df.groupby(['CustomerID', 'Quarter']).size().unstack(fill_value=0)
    quarterly.columns = [f'Q{col}_Transactions' for col in quarterly.columns]
    quarterly = quarterly.reset_index()
    
    seasonal = seasonal.merge(quarterly, on='CustomerID', how='left')
    
    # Calculate preferred quarter (quarter with most transactions)
    quarterly_counts = transactions_df.groupby(['CustomerID', 'Quarter']).size().reset_index(name='Count')
    preferred_quarter = quarterly_counts.loc[
        quarterly_counts.groupby('CustomerID')['Count'].idxmax()
    ][['CustomerID', 'Quarter']]
    preferred_quarter.columns = ['CustomerID', 'PreferredQuarter']
    
    seasonal = seasonal.merge(preferred_quarter, on='CustomerID', how='left')
    
    logger.info(f"Seasonal features calculated for {len(seasonal)} customers")
    
    return seasonal


def create_customer_feature_matrix(transactions_df: pd.DataFrame,
                                   loyalty_df: pd.DataFrame,
                                   customers_df: pd.DataFrame,
                                   products_df: pd.DataFrame,
                                   surveys_df: Optional[pd.DataFrame] = None,
                                   reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Create comprehensive customer feature matrix combining all features.
    
    Args:
        transactions_df: Transaction data
        loyalty_df: Loyalty data
        customers_df: Customer data
        products_df: Product data
        surveys_df: Optional survey data
        reference_date: Reference date for calculations
        
    Returns:
        Complete feature matrix DataFrame
    """
    if reference_date is None:
        reference_date = transactions_df['TransactionDate'].max()
    
    logger.info("Creating comprehensive customer feature matrix")
    
    # Start with customer base
    feature_matrix = customers_df[['CustomerID']].copy()
    
    # Calculate all feature sets
    logger.info("  - Calculating RFM features...")
    rfm_features = calculate_rfm_features(transactions_df, reference_date)
    feature_matrix = feature_matrix.merge(rfm_features, on='CustomerID', how='left')
    
    logger.info("  - Calculating frequency by period...")
    frequency_features = calculate_frequency_by_period(transactions_df, reference_date)
    feature_matrix = feature_matrix.merge(frequency_features, on='CustomerID', how='left', suffixes=('', '_freq'))
    
    logger.info("  - Calculating loyalty features...")
    loyalty_features = calculate_loyalty_features(loyalty_df, customers_df, reference_date)
    feature_matrix = feature_matrix.merge(loyalty_features, on='CustomerID', how='left')
    
    logger.info("  - Calculating engagement features...")
    engagement_features = calculate_engagement_features(transactions_df, products_df)
    feature_matrix = feature_matrix.merge(engagement_features, on='CustomerID', how='left')
    
    logger.info("  - Calculating behavioral features...")
    behavioral_features = calculate_behavioral_features(transactions_df, reference_date)
    feature_matrix = feature_matrix.merge(behavioral_features, on='CustomerID', how='left')
    
    logger.info("  - Calculating seasonal features...")
    seasonal_features = calculate_seasonal_features(transactions_df)
    feature_matrix = feature_matrix.merge(seasonal_features, on='CustomerID', how='left')
    
    # Add survey features if available
    if surveys_df is not None and len(surveys_df) > 0:
        logger.info("  - Adding survey features...")
        survey_features = surveys_df.groupby('CustomerID').agg({
            'SatisfactionScore': 'mean',
            'NPSScore': 'mean',
            'WouldRecommend': lambda x: (x == 'Yes').sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        survey_features.columns = ['CustomerID', 'AvgSatisfactionScore', 'AvgNPSScore', 'RecommendationRate']
        feature_matrix = feature_matrix.merge(survey_features, on='CustomerID', how='left')
    
    # Fill missing values
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    feature_matrix[numeric_cols] = feature_matrix[numeric_cols].fillna(0)
    
    logger.info(f"Feature matrix created with {len(feature_matrix)} customers and {len(feature_matrix.columns)} features")
    
    return feature_matrix

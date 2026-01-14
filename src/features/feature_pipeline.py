"""
Feature Engineering Pipeline

This module orchestrates the complete feature engineering process:
1. Calculate all customer and product features
2. Combine into feature matrices
3. Handle feature scaling and selection
4. Prepare features for machine learning
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from sklearn.preprocessing import StandardScaler
import logging

from . import customer_features, product_features

logger = logging.getLogger(__name__)


def create_feature_matrix(transactions_df: pd.DataFrame,
                         loyalty_df: pd.DataFrame,
                         customers_df: pd.DataFrame,
                         products_df: pd.DataFrame,
                         surveys_df: Optional[pd.DataFrame] = None,
                         reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Create comprehensive feature matrix combining all customer features.
    
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
    logger.info("=" * 60)
    logger.info("Creating Feature Matrix")
    logger.info("=" * 60)
    
    feature_matrix = customer_features.create_customer_feature_matrix(
        transactions_df=transactions_df,
        loyalty_df=loyalty_df,
        customers_df=customers_df,
        products_df=products_df,
        surveys_df=surveys_df,
        reference_date=reference_date
    )
    
    logger.info(f"\nFeature matrix created:")
    logger.info(f"  - Customers: {len(feature_matrix):,}")
    logger.info(f"  - Features: {len(feature_matrix.columns):,}")
    logger.info(f"  - Memory usage: {feature_matrix.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return feature_matrix


def identify_highly_correlated_features(feature_matrix: pd.DataFrame,
                                       threshold: float = 0.95) -> List[str]:
    """
    Identify highly correlated features for removal.
    
    Args:
        feature_matrix: Feature matrix DataFrame
        threshold: Correlation threshold (default: 0.95)
        
    Returns:
        List of feature names to remove
    """
    logger.info(f"Identifying highly correlated features (threshold: {threshold})")
    
    # Select only numeric columns
    numeric_features = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'CustomerID' in numeric_features:
        numeric_features.remove('CustomerID')
    
    if len(numeric_features) == 0:
        logger.warning("No numeric features found for correlation analysis")
        return []
    
    # Calculate correlation matrix
    corr_matrix = feature_matrix[numeric_features].corr().abs()
    
    # Find pairs with high correlation
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    if to_drop:
        logger.info(f"Found {len(to_drop)} highly correlated features to remove: {to_drop}")
    else:
        logger.info("No highly correlated features found")
    
    return to_drop


def remove_highly_correlated_features(feature_matrix: pd.DataFrame,
                                     threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove highly correlated features from feature matrix.
    
    Args:
        feature_matrix: Feature matrix DataFrame
        threshold: Correlation threshold
        
    Returns:
        Feature matrix with correlated features removed
    """
    to_drop = identify_highly_correlated_features(feature_matrix, threshold)
    
    if to_drop:
        feature_matrix = feature_matrix.drop(columns=to_drop)
        logger.info(f"Removed {len(to_drop)} highly correlated features")
    
    return feature_matrix


def scale_features(feature_matrix: pd.DataFrame,
                  exclude_columns: Optional[List[str]] = None,
                  scaler: Optional[StandardScaler] = None) -> tuple:
    """
    Scale features using StandardScaler.
    
    Args:
        feature_matrix: Feature matrix DataFrame
        exclude_columns: Columns to exclude from scaling (e.g., CustomerID, categorical)
        scaler: Optional pre-fitted scaler (if None, fits new scaler)
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    if exclude_columns is None:
        exclude_columns = ['CustomerID']
    
    logger.info("Scaling features")
    
    # Identify columns to scale
    columns_to_scale = [col for col in feature_matrix.columns 
                        if col not in exclude_columns and 
                        feature_matrix[col].dtype in [np.int64, np.int32, np.float64, np.float32]]
    
    if len(columns_to_scale) == 0:
        logger.warning("No numeric features found to scale")
        return feature_matrix, None
    
    # Separate columns to keep and scale
    keep_columns = [col for col in feature_matrix.columns if col not in columns_to_scale]
    scaled_data = feature_matrix[columns_to_scale].copy()
    
    # Handle infinite and NaN values
    scaled_data = scaled_data.replace([np.inf, -np.inf], np.nan)
    scaled_data = scaled_data.fillna(scaled_data.median())
    
    # Fit or use provided scaler
    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(scaled_data)
    else:
        scaled_values = scaler.transform(scaled_data)
    
    # Create scaled DataFrame
    scaled_df = pd.DataFrame(
        scaled_values,
        columns=columns_to_scale,
        index=scaled_data.index
    )
    
    # Combine with non-scaled columns
    if keep_columns:
        result = pd.concat([feature_matrix[keep_columns], scaled_df], axis=1)
    else:
        result = scaled_df
    
    logger.info(f"Scaled {len(columns_to_scale)} features")
    
    return result, scaler


def prepare_features_for_ml(feature_matrix: pd.DataFrame,
                           target_column: Optional[str] = None,
                           remove_correlated: bool = True,
                           scale_features_flag: bool = True,
                           exclude_from_scaling: Optional[List[str]] = None) -> Dict:
    """
    Prepare features for machine learning.
    
    Args:
        feature_matrix: Feature matrix DataFrame
        target_column: Optional target column to separate
        remove_correlated: Whether to remove highly correlated features
        scale_features_flag: Whether to scale features
        exclude_from_scaling: Columns to exclude from scaling
        
    Returns:
        Dictionary with prepared features and metadata
    """
    logger.info("=" * 60)
    logger.info("Preparing Features for Machine Learning")
    logger.info("=" * 60)
    
    result = {}
    
    # Separate target if provided
    if target_column and target_column in feature_matrix.columns:
        result['target'] = feature_matrix[target_column].copy()
        feature_matrix = feature_matrix.drop(columns=[target_column])
        logger.info(f"Separated target column: {target_column}")
    
    # Remove highly correlated features
    if remove_correlated:
        feature_matrix = remove_highly_correlated_features(feature_matrix)
    
    # Scale features
    scaler = None
    if scale_features_flag:
        if exclude_from_scaling is None:
            exclude_from_scaling = ['CustomerID']
        feature_matrix, scaler = scale_features(feature_matrix, exclude_from_scaling)
        result['scaler'] = scaler
    
    result['features'] = feature_matrix
    result['feature_names'] = [col for col in feature_matrix.columns 
                              if col not in ['CustomerID']]
    
    logger.info(f"\nFeatures prepared:")
    logger.info(f"  - Total features: {len(result['feature_names'])}")
    logger.info(f"  - Samples: {len(feature_matrix):,}")
    
    return result


def get_feature_summary(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for all features.
    
    Args:
        feature_matrix: Feature matrix DataFrame
        
    Returns:
        DataFrame with feature summary statistics
    """
    numeric_features = feature_matrix.select_dtypes(include=[np.number])
    
    summary = pd.DataFrame({
        'Feature': numeric_features.columns,
        'Mean': numeric_features.mean(),
        'Std': numeric_features.std(),
        'Min': numeric_features.min(),
        'Max': numeric_features.max(),
        'Missing': numeric_features.isna().sum(),
        'MissingPct': (numeric_features.isna().sum() / len(numeric_features)) * 100
    })
    
    return summary.sort_values('Feature')

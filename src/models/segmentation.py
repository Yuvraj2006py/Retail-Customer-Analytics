"""
Customer Segmentation Module

This module implements K-Means clustering for customer segmentation
with multiple segmentation approaches and comprehensive segment profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def select_features_for_clustering(feature_matrix: pd.DataFrame,
                                   feature_types: List[str] = ['rfm', 'loyalty', 'engagement', 'behavioral'],
                                   exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Select features for clustering based on feature types.
    
    Args:
        feature_matrix: Complete feature matrix
        feature_types: List of feature types to include ('rfm', 'loyalty', 'engagement', 'behavioral', 'seasonal')
        exclude_columns: Columns to exclude from clustering
        
    Returns:
        DataFrame with selected features
    """
    if exclude_columns is None:
        exclude_columns = ['CustomerID']
    
    logger.info(f"Selecting features for clustering: {feature_types}")
    
    # Define feature groups
    feature_groups = {
        'rfm': ['Recency', 'Frequency', 'TotalSpend', 'TotalSpend_x', 'TotalSpend_y',
                'RecencyScore', 'FrequencyScore', 'MonetaryScore', 'RFMScore',
                'LifetimeValue', 'AvgTransactionValue', 'CustomerLifetime'],
        'loyalty': ['CurrentPointsBalance', 'TotalPointsEarned', 'RedemptionRate',
                   'TierNumeric', 'DaysSinceEnrollment', 'PointsPerDollarRatio', 'TierChanges'],
        'engagement': ['DaysActive', 'ActiveMonths', 'UniqueProducts', 'UniqueCategories',
                      'AvgBasketSize', 'PurchaseVelocity', 'ProductDiversity'],
        'behavioral': ['FrequencyTrend', 'SpendingTrend', 'BasketSizeTrend',
                      'RecentFrequency', 'RecentSpend', 'HistoricalFrequency', 'HistoricalSpend'],
        'seasonal': ['HolidayActivityRatio', 'PreferredQuarter'],
        'frequency': ['Frequency_30d', 'Frequency_60d', 'Frequency_90d', 'Frequency_365d',
                     'Spend_30d', 'Spend_60d', 'Spend_90d', 'Spend_365d']
    }
    
    # Select features
    selected_features = []
    for feature_type in feature_types:
        if feature_type in feature_groups:
            selected_features.extend(feature_groups[feature_type])
    
    # Get available features (some may not exist in the matrix)
    available_features = [f for f in selected_features if f in feature_matrix.columns]
    
    # Remove excluded columns
    available_features = [f for f in available_features if f not in exclude_columns]
    
    # Select only numeric features
    numeric_features = feature_matrix[available_features].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) == 0:
        raise ValueError("No numeric features found for clustering")
    
    logger.info(f"Selected {len(numeric_features)} features for clustering")
    
    return feature_matrix[['CustomerID'] + numeric_features].copy()


def prepare_features_for_clustering(feature_matrix: pd.DataFrame,
                                   use_pca: bool = False,
                                   pca_variance: float = 0.95) -> Tuple[pd.DataFrame, Optional[StandardScaler], Optional[PCA]]:
    """
    Prepare and scale features for clustering.
    
    Args:
        feature_matrix: Feature matrix with CustomerID and features
        use_pca: Whether to apply PCA dimensionality reduction
        pca_variance: Variance to retain if using PCA
        
    Returns:
        Tuple of (prepared DataFrame, fitted scaler, fitted PCA transformer)
    """
    logger.info("Preparing features for clustering")
    
    # Separate CustomerID and features
    customer_ids = feature_matrix['CustomerID'].copy()
    features = feature_matrix.drop(columns=['CustomerID'])
    
    # Handle missing values
    features = features.fillna(features.median())
    
    # Handle infinite values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(
        features_scaled,
        columns=features.columns,
        index=features.index
    )
    
    # Apply PCA if requested
    pca = None
    if use_pca:
        logger.info(f"Applying PCA to retain {pca_variance*100}% variance")
        pca = PCA(n_components=pca_variance)
        features_scaled = pca.fit_transform(features_scaled)
        
        # Create DataFrame with PCA components
        pca_columns = [f'PC{i+1}' for i in range(features_scaled.shape[1])]
        features_scaled_df = pd.DataFrame(
            features_scaled,
            columns=pca_columns,
            index=features.index
        )
        
        explained_variance = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA retained {explained_variance*100:.2f}% variance with {len(pca_columns)} components")
    
    # Combine with CustomerID
    result = pd.concat([customer_ids.reset_index(drop=True), features_scaled_df.reset_index(drop=True)], axis=1)
    
    return result, scaler, pca


def find_optimal_k_clusters(feature_matrix: pd.DataFrame,
                            k_range: range = range(3, 9),
                            random_state: int = 42) -> Dict[int, Dict[str, float]]:
    """
    Find optimal number of clusters using elbow method and silhouette score.
    
    Args:
        feature_matrix: Prepared feature matrix (scaled, without CustomerID)
        k_range: Range of K values to test
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with K values and their metrics
    """
    logger.info(f"Finding optimal K in range {k_range.start} to {k_range.stop-1}")
    
    # Extract features (without CustomerID)
    if 'CustomerID' in feature_matrix.columns:
        features = feature_matrix.drop(columns=['CustomerID']).values
    else:
        features = feature_matrix.values
    
    results = {}
    wcss_scores = []  # Within-cluster sum of squares
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(features)
        
        # Calculate WCSS (inertia)
        wcss = kmeans.inertia_
        wcss_scores.append(wcss)
        
        # Calculate silhouette score
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters
            silhouette = silhouette_score(features, labels)
            silhouette_scores.append(silhouette)
        else:
            silhouette = -1
            silhouette_scores.append(-1)
        
        results[k] = {
            'wcss': wcss,
            'silhouette': silhouette
        }
        
        logger.info(f"  K={k}: WCSS={wcss:.2f}, Silhouette={silhouette:.4f}")
    
    # Find optimal K (highest silhouette score)
    optimal_k = max(results.keys(), key=lambda k: results[k]['silhouette'])
    logger.info(f"Optimal K: {optimal_k} (silhouette score: {results[optimal_k]['silhouette']:.4f})")
    
    return results


def perform_kmeans_clustering(feature_matrix: pd.DataFrame,
                              n_clusters: int = 5,
                              random_state: int = 42,
                              max_iter: int = 300,
                              n_init: int = 10) -> Tuple[KMeans, pd.Series]:
    """
    Perform K-Means clustering on feature matrix.
    
    Args:
        feature_matrix: Prepared feature matrix
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        max_iter: Maximum iterations
        n_init: Number of initializations
        
    Returns:
        Tuple of (fitted KMeans model, cluster labels)
    """
    logger.info(f"Performing K-Means clustering with {n_clusters} clusters")
    
    # Extract features
    if 'CustomerID' in feature_matrix.columns:
        features = feature_matrix.drop(columns=['CustomerID']).values
        customer_ids = feature_matrix['CustomerID'].values
    else:
        features = feature_matrix.values
        customer_ids = None
    
    # Fit K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
        init='k-means++'
    )
    
    labels = kmeans.fit_predict(features)
    
    # Create labels Series with CustomerID index if available
    if customer_ids is not None:
        labels_series = pd.Series(labels, index=customer_ids, name='Segment')
    else:
        labels_series = pd.Series(labels, name='Segment')
    
    logger.info(f"Clustering complete. Cluster distribution:")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        logger.info(f"  Cluster {cluster}: {count} customers ({count/len(labels)*100:.1f}%)")
    
    return kmeans, labels_series


def profile_segments(feature_matrix: pd.DataFrame,
                    segment_labels: pd.Series,
                    transactions_df: Optional[pd.DataFrame] = None,
                    loyalty_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create detailed profiles for each customer segment.
    
    Args:
        feature_matrix: Original feature matrix
        segment_labels: Cluster labels for each customer
        transactions_df: Optional transaction data for additional metrics
        loyalty_df: Optional loyalty data for additional metrics
        
    Returns:
        DataFrame with segment profiles
    """
    logger.info("Profiling customer segments")
    
    # Combine features with segment labels
    feature_with_segments = feature_matrix.copy()
    feature_with_segments['Segment'] = segment_labels.values
    
    # Group by segment and calculate statistics
    numeric_features = feature_with_segments.select_dtypes(include=[np.number]).columns.tolist()
    if 'Segment' in numeric_features:
        numeric_features.remove('Segment')
    
    segment_profiles = feature_with_segments.groupby('Segment')[numeric_features].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).round(2)
    
    # Flatten column names
    segment_profiles.columns = ['_'.join(col).strip() for col in segment_profiles.columns.values]
    
    # Calculate segment sizes
    segment_sizes = feature_with_segments.groupby('Segment').size()
    segment_profiles['SegmentSize'] = segment_sizes
    segment_profiles['SegmentPercentage'] = (segment_sizes / len(feature_with_segments) * 100).round(2)
    
    # Key metrics summary
    key_metrics = {}
    for segment in segment_profiles.index:
        segment_data = feature_with_segments[feature_with_segments['Segment'] == segment]
        
        # Helper function to safely calculate mean
        def safe_mean(series):
            if series is None or len(series) == 0:
                return None
            try:
                # Convert to numeric if needed
                if series.dtype == 'category':
                    return series.astype('float64').mean()
                return series.mean()
            except:
                return None
        
        # Get TotalSpend column (handle different names)
        total_spend_col = None
        for col in ['TotalSpend', 'TotalSpend_x', 'TotalSpend_y']:
            if col in segment_data.columns:
                total_spend_col = segment_data[col]
                break
        
        key_metrics[segment] = {
            'AvgRecency': safe_mean(segment_data['Recency']) if 'Recency' in segment_data.columns else None,
            'AvgFrequency': safe_mean(segment_data['Frequency']) if 'Frequency' in segment_data.columns else None,
            'AvgTotalSpend': safe_mean(total_spend_col) if total_spend_col is not None else None,
            'AvgLifetimeValue': safe_mean(segment_data['LifetimeValue']) if 'LifetimeValue' in segment_data.columns else None,
            'AvgPointsBalance': safe_mean(segment_data['CurrentPointsBalance']) if 'CurrentPointsBalance' in segment_data.columns else None,
            'AvgTier': safe_mean(segment_data['TierNumeric']) if 'TierNumeric' in segment_data.columns else None,
        }
    
    # Add key metrics to profiles
    for metric, values in key_metrics.items():
        for key, value in values.items():
            if value is not None:
                segment_profiles.loc[metric, key] = value
    
    logger.info(f"Segment profiles created for {len(segment_profiles)} segments")
    
    return segment_profiles.reset_index()


def assign_segment_names(segment_profiles: pd.DataFrame,
                         feature_matrix: pd.DataFrame,
                         segment_labels: pd.Series) -> Dict[int, str]:
    """
    Assign meaningful names to segments based on their characteristics.
    
    Args:
        segment_profiles: Segment profile DataFrame
        feature_matrix: Feature matrix
        segment_labels: Segment labels
        
    Returns:
        Dictionary mapping segment numbers to names
    """
    logger.info("Assigning segment names based on characteristics")
    
    feature_with_segments = feature_matrix.copy()
    feature_with_segments['Segment'] = segment_labels.values
    
    segment_names = {}
    
    for segment in segment_profiles['Segment'].unique():
        segment_data = feature_with_segments[feature_with_segments['Segment'] == segment]
        
        # Helper function to safely get mean
        def safe_mean(series):
            if series is None or len(series) == 0:
                return 0
            try:
                if hasattr(series, 'dtype') and series.dtype == 'category':
                    return series.astype('float64').mean()
                return float(series.mean())
            except:
                return 0
        
        # Get key metrics
        recency = safe_mean(segment_data['Recency']) if 'Recency' in segment_data.columns else 999
        frequency = safe_mean(segment_data['Frequency']) if 'Frequency' in segment_data.columns else 0
        
        # Get TotalSpend column (handle different names)
        total_spend = 0
        for col in ['TotalSpend', 'TotalSpend_x', 'TotalSpend_y']:
            if col in segment_data.columns:
                total_spend = safe_mean(segment_data[col])
                break
        
        lifetime_value = safe_mean(segment_data['LifetimeValue']) if 'LifetimeValue' in segment_data.columns else 0
        points_balance = safe_mean(segment_data['CurrentPointsBalance']) if 'CurrentPointsBalance' in segment_data.columns else 0
        tier = safe_mean(segment_data['TierNumeric']) if 'TierNumeric' in segment_data.columns else 1
        
        # Determine segment name based on characteristics
        if recency <= 30 and frequency >= 20 and total_spend >= 5000:
            name = "Champions"
        elif recency <= 60 and frequency >= 10 and total_spend >= 2000:
            name = "Loyal Customers"
        elif recency <= 90 and frequency >= 5:
            name = "Potential Loyalists"
        elif recency <= 30 and frequency <= 5:
            name = "New Customers"
        elif recency > 90 and frequency >= 10:
            name = "At Risk"
        elif recency > 180:
            name = "Lost"
        elif tier >= 3 and points_balance >= 25000:
            name = "High-Value Loyal"
        elif lifetime_value >= 10000:
            name = "High-Value"
        elif recency > 90 and recency <= 180 and frequency >= 5 and frequency < 10 and total_spend >= 2000:
            name = "Dormant Customers"
        elif recency > 90 and frequency < 10 and total_spend >= 2000:
            name = "Moderate Customers"
        elif recency <= 120 and frequency >= 5 and total_spend >= 1000:
            name = "Regular Customers"
        elif frequency >= 5:
            name = "Occasional Customers"
        else:
            name = "Low Engagement"
        
        segment_names[segment] = name
        logger.info(f"  Segment {segment}: {name}")
    
    return segment_names


def perform_customer_segmentation(feature_matrix: pd.DataFrame,
                                n_clusters: Optional[int] = None,
                                feature_types: List[str] = ['rfm', 'loyalty', 'engagement', 'behavioral'],
                                find_optimal_k: bool = True,
                                k_range: range = range(3, 9),
                                use_pca: bool = False,
                                random_state: int = 42) -> Dict:
    """
    Complete customer segmentation pipeline.
    
    Args:
        feature_matrix: Complete customer feature matrix
        n_clusters: Number of clusters (if None, will find optimal)
        feature_types: Feature types to use for clustering
        find_optimal_k: Whether to find optimal K
        k_range: Range of K values to test
        use_pca: Whether to use PCA
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with segmentation results
    """
    logger.info("=" * 60)
    logger.info("Customer Segmentation Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Select features
    logger.info("\nStep 1: Selecting features...")
    selected_features = select_features_for_clustering(feature_matrix, feature_types)
    
    # Step 2: Prepare features
    logger.info("\nStep 2: Preparing features...")
    prepared_features, scaler, pca = prepare_features_for_clustering(selected_features, use_pca)
    
    # Step 3: Find optimal K if needed
    if find_optimal_k and n_clusters is None:
        logger.info("\nStep 3: Finding optimal number of clusters...")
        k_metrics = find_optimal_k_clusters(prepared_features, k_range, random_state)
        n_clusters = max(k_metrics.keys(), key=lambda k: k_metrics[k]['silhouette'])
        logger.info(f"Selected optimal K: {n_clusters}")
    elif n_clusters is None:
        n_clusters = 5  # Default
        logger.info(f"Using default K: {n_clusters}")
    
    # Step 4: Perform clustering
    logger.info(f"\nStep 4: Performing K-Means clustering with {n_clusters} clusters...")
    kmeans_model, segment_labels = perform_kmeans_clustering(
        prepared_features,
        n_clusters=n_clusters,
        random_state=random_state
    )
    
    # Step 5: Profile segments
    logger.info("\nStep 5: Profiling segments...")
    segment_profiles = profile_segments(feature_matrix, segment_labels)
    
    # Step 6: Assign segment names
    logger.info("\nStep 6: Assigning segment names...")
    segment_names = assign_segment_names(segment_profiles, feature_matrix, segment_labels)
    
    # Create results dictionary
    results = {
        'model': kmeans_model,
        'labels': segment_labels,
        'profiles': segment_profiles,
        'segment_names': segment_names,
        'n_clusters': n_clusters,
        'scaler': scaler,
        'pca': pca,
        'feature_matrix': feature_matrix,
        'selected_features': selected_features.columns.tolist()
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("Segmentation Complete!")
    logger.info("=" * 60)
    logger.info(f"\nSegments created:")
    for segment_num, segment_name in segment_names.items():
        count = (segment_labels == segment_num).sum()
        pct = count / len(segment_labels) * 100
        logger.info(f"  {segment_name} (Segment {segment_num}): {count} customers ({pct:.1f}%)")
    
    return results


def save_segmentation_results(results: Dict, output_path: str) -> None:
    """
    Save segmentation results to files.
    
    Args:
        results: Segmentation results dictionary
        output_path: Output directory path
    """
    from pathlib import Path
    import joblib
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving segmentation results to {output_dir}")
    
    # Save model
    joblib.dump(results['model'], output_dir / 'kmeans_model.pkl')
    
    # Save scaler and PCA if available
    if results['scaler'] is not None:
        joblib.dump(results['scaler'], output_dir / 'scaler.pkl')
    if results['pca'] is not None:
        joblib.dump(results['pca'], output_dir / 'pca.pkl')
    
    # Save segment labels
    results['labels'].to_csv(output_dir / 'segment_labels.csv')
    
    # Save segment profiles
    results['profiles'].to_csv(output_dir / 'segment_profiles.csv', index=False)
    
    # Save segment names mapping
    import json
    with open(output_dir / 'segment_names.json', 'w') as f:
        json.dump(results['segment_names'], f, indent=2)
    
    logger.info("Segmentation results saved successfully")

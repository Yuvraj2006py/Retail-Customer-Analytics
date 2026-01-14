"""
Comprehensive unit tests for customer segmentation module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import segmentation


class TestSegmentation:
    """Test customer segmentation functions."""
    
    @pytest.fixture
    def sample_feature_matrix(self):
        """Create sample feature matrix."""
        np.random.seed(42)
        n_customers = 100
        
        return pd.DataFrame({
            'CustomerID': [f'CUST_{i:05d}' for i in range(1, n_customers + 1)],
            # RFM features
            'Recency': np.random.randint(0, 365, n_customers),
            'Frequency': np.random.randint(1, 100, n_customers),
            'TotalSpend': np.random.uniform(100, 10000, n_customers),
            'RecencyScore': np.random.randint(1, 6, n_customers),
            'FrequencyScore': np.random.randint(1, 6, n_customers),
            'MonetaryScore': np.random.randint(1, 6, n_customers),
            'RFMScore': np.random.randint(111, 555, n_customers),
            'LifetimeValue': np.random.uniform(500, 15000, n_customers),
            'AvgTransactionValue': np.random.uniform(10, 500, n_customers),
            # Loyalty features
            'CurrentPointsBalance': np.random.randint(0, 50000, n_customers),
            'TotalPointsEarned': np.random.randint(0, 100000, n_customers),
            'RedemptionRate': np.random.uniform(0, 1, n_customers),
            'TierNumeric': np.random.randint(1, 5, n_customers),
            'DaysSinceEnrollment': np.random.randint(30, 1000, n_customers),
            # Engagement features
            'DaysActive': np.random.randint(1, 365, n_customers),
            'ActiveMonths': np.random.randint(1, 24, n_customers),
            'UniqueProducts': np.random.randint(1, 50, n_customers),
            'AvgBasketSize': np.random.uniform(1, 10, n_customers),
            'PurchaseVelocity': np.random.uniform(0.5, 10, n_customers),
            # Behavioral features
            'FrequencyTrend': np.random.uniform(-0.5, 0.5, n_customers),
            'SpendingTrend': np.random.uniform(-0.5, 0.5, n_customers),
            'RecentFrequency': np.random.randint(0, 20, n_customers),
            'RecentSpend': np.random.uniform(0, 5000, n_customers),
        })
    
    def test_select_features_for_clustering(self, sample_feature_matrix):
        """Test feature selection for clustering."""
        selected = segmentation.select_features_for_clustering(
            sample_feature_matrix,
            feature_types=['rfm', 'loyalty']
        )
        
        assert isinstance(selected, pd.DataFrame)
        assert 'CustomerID' in selected.columns
        assert 'Recency' in selected.columns
        assert 'Frequency' in selected.columns
        assert 'CurrentPointsBalance' in selected.columns
        assert len(selected.columns) > 1
    
    def test_prepare_features_for_clustering(self, sample_feature_matrix):
        """Test feature preparation."""
        selected = segmentation.select_features_for_clustering(sample_feature_matrix)
        prepared, scaler, pca = segmentation.prepare_features_for_clustering(selected)
        
        assert isinstance(prepared, pd.DataFrame)
        assert 'CustomerID' in prepared.columns
        assert scaler is not None
        assert pca is None  # PCA not used by default
        
        # Check that features are scaled (mean ~0, std ~1)
        feature_cols = [col for col in prepared.columns if col != 'CustomerID']
        for col in feature_cols[:5]:  # Check first 5 features
            assert abs(prepared[col].mean()) < 0.1
            assert abs(prepared[col].std() - 1.0) < 0.2
    
    def test_prepare_features_with_pca(self, sample_feature_matrix):
        """Test feature preparation with PCA."""
        selected = segmentation.select_features_for_clustering(sample_feature_matrix)
        prepared, scaler, pca = segmentation.prepare_features_for_clustering(
            selected, use_pca=True, pca_variance=0.95
        )
        
        assert isinstance(prepared, pd.DataFrame)
        assert scaler is not None
        assert pca is not None
        assert 'PC1' in prepared.columns
    
    def test_find_optimal_k(self, sample_feature_matrix):
        """Test optimal K finding."""
        selected = segmentation.select_features_for_clustering(sample_feature_matrix)
        prepared, _, _ = segmentation.prepare_features_for_clustering(selected)
        
        k_metrics = segmentation.find_optimal_k_clusters(prepared, k_range=range(3, 6))
        
        assert isinstance(k_metrics, dict)
        assert 3 in k_metrics
        assert 4 in k_metrics
        assert 5 in k_metrics
        
        # Verify metrics structure
        for k, metrics in k_metrics.items():
            assert 'wcss' in metrics
            assert 'silhouette' in metrics
            assert metrics['wcss'] > 0
            assert -1 <= metrics['silhouette'] <= 1
    
    def test_perform_kmeans_clustering(self, sample_feature_matrix):
        """Test K-Means clustering."""
        selected = segmentation.select_features_for_clustering(sample_feature_matrix)
        prepared, _, _ = segmentation.prepare_features_for_clustering(selected)
        
        kmeans, labels = segmentation.perform_kmeans_clustering(prepared, n_clusters=5)
        
        assert isinstance(kmeans, KMeans)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(prepared)
        assert labels.nunique() == 5  # Should have 5 clusters
        assert all(labels >= 0) and all(labels < 5)
    
    def test_profile_segments(self, sample_feature_matrix):
        """Test segment profiling."""
        selected = segmentation.select_features_for_clustering(sample_feature_matrix)
        prepared, _, _ = segmentation.prepare_features_for_clustering(selected)
        _, labels = segmentation.perform_kmeans_clustering(prepared, n_clusters=5)
        
        profiles = segmentation.profile_segments(sample_feature_matrix, labels)
        
        assert isinstance(profiles, pd.DataFrame)
        assert 'Segment' in profiles.columns
        assert len(profiles) == 5  # 5 segments
        assert 'SegmentSize' in profiles.columns
        assert 'SegmentPercentage' in profiles.columns
    
    def test_assign_segment_names(self, sample_feature_matrix):
        """Test segment name assignment."""
        selected = segmentation.select_features_for_clustering(sample_feature_matrix)
        prepared, _, _ = segmentation.prepare_features_for_clustering(selected)
        _, labels = segmentation.perform_kmeans_clustering(prepared, n_clusters=5)
        profiles = segmentation.profile_segments(sample_feature_matrix, labels)
        
        segment_names = segmentation.assign_segment_names(profiles, sample_feature_matrix, labels)
        
        assert isinstance(segment_names, dict)
        assert len(segment_names) == 5
        assert all(isinstance(name, str) for name in segment_names.values())
    
    def test_perform_customer_segmentation_complete(self, sample_feature_matrix):
        """Test complete segmentation pipeline."""
        results = segmentation.perform_customer_segmentation(
            sample_feature_matrix,
            n_clusters=5,
            find_optimal_k=False
        )
        
        assert isinstance(results, dict)
        assert 'model' in results
        assert 'labels' in results
        assert 'profiles' in results
        assert 'segment_names' in results
        assert 'n_clusters' in results
        assert results['n_clusters'] == 5
        assert len(results['labels']) == len(sample_feature_matrix)
        assert len(results['profiles']) == 5
    
    def test_perform_customer_segmentation_with_optimal_k(self, sample_feature_matrix):
        """Test segmentation with optimal K finding."""
        results = segmentation.perform_customer_segmentation(
            sample_feature_matrix,
            find_optimal_k=True,
            k_range=range(3, 6)
        )
        
        assert isinstance(results, dict)
        assert 'model' in results
        assert 'labels' in results
        assert results['n_clusters'] in [3, 4, 5]
        assert len(results['labels']) == len(sample_feature_matrix)
    
    def test_segmentation_with_different_feature_types(self, sample_feature_matrix):
        """Test segmentation with different feature type combinations."""
        # Test with only RFM features
        results_rfm = segmentation.perform_customer_segmentation(
            sample_feature_matrix,
            feature_types=['rfm'],
            n_clusters=3,
            find_optimal_k=False
        )
        assert len(results_rfm['labels']) == len(sample_feature_matrix)
        
        # Test with RFM + Loyalty
        results_combined = segmentation.perform_customer_segmentation(
            sample_feature_matrix,
            feature_types=['rfm', 'loyalty'],
            n_clusters=3,
            find_optimal_k=False
        )
        assert len(results_combined['labels']) == len(sample_feature_matrix)
    
    def test_segmentation_handles_missing_values(self):
        """Test that segmentation handles missing values correctly."""
        # Create feature matrix with missing values
        feature_matrix = pd.DataFrame({
            'CustomerID': ['CUST_001', 'CUST_002', 'CUST_003'],
            'Recency': [10, 50, np.nan],
            'Frequency': [20, 15, 10],
            'TotalSpend': [5000, np.nan, 2000],
            'CurrentPointsBalance': [10000, 5000, np.nan]
        })
        
        results = segmentation.perform_customer_segmentation(
            feature_matrix,
            n_clusters=2,
            find_optimal_k=False
        )
        
        assert len(results['labels']) == 3
        assert all(results['labels'] >= 0)
    
    def test_segmentation_handles_infinite_values(self):
        """Test that segmentation handles infinite values correctly."""
        feature_matrix = pd.DataFrame({
            'CustomerID': ['CUST_001', 'CUST_002', 'CUST_003'],
            'Recency': [10, 50, 100],
            'Frequency': [20, 15, 10],
            'TotalSpend': [5000, np.inf, 2000],
            'CurrentPointsBalance': [10000, 5000, -np.inf]
        })
        
        results = segmentation.perform_customer_segmentation(
            feature_matrix,
            n_clusters=2,
            find_optimal_k=False
        )
        
        assert len(results['labels']) == 3
        assert all(results['labels'] >= 0)
    
    def test_segmentation_with_pca(self, sample_feature_matrix):
        """Test segmentation with PCA dimensionality reduction."""
        results = segmentation.perform_customer_segmentation(
            sample_feature_matrix,
            n_clusters=5,
            use_pca=True,
            find_optimal_k=False
        )
        
        assert results['pca'] is not None
        assert len(results['labels']) == len(sample_feature_matrix)
    
    def test_segment_profiles_contain_key_metrics(self, sample_feature_matrix):
        """Test that segment profiles contain expected metrics."""
        results = segmentation.perform_customer_segmentation(
            sample_feature_matrix,
            n_clusters=5,
            find_optimal_k=False
        )
        
        profiles = results['profiles']
        assert 'Segment' in profiles.columns
        assert 'SegmentSize' in profiles.columns
        assert 'SegmentPercentage' in profiles.columns
        
        # Verify segment percentages sum to ~100%
        total_pct = profiles['SegmentPercentage'].sum()
        assert abs(total_pct - 100.0) < 1.0
    
    def test_segmentation_reproducibility(self, sample_feature_matrix):
        """Test that segmentation is reproducible with same random_state."""
        results1 = segmentation.perform_customer_segmentation(
            sample_feature_matrix,
            n_clusters=5,
            random_state=42,
            find_optimal_k=False
        )
        
        results2 = segmentation.perform_customer_segmentation(
            sample_feature_matrix,
            n_clusters=5,
            random_state=42,
            find_optimal_k=False
        )
        
        # Labels should be identical
        assert results1['labels'].equals(results2['labels'])

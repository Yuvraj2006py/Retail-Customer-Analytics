"""
Unit tests for data generation module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.generate_synthetic_data import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test cases for SyntheticDataGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return SyntheticDataGenerator("config/data_generation_config.yaml")
    
    def test_config_loading(self, generator):
        """Test that configuration loads correctly."""
        assert generator.config is not None
        assert 'data_generation' in generator.config
        assert generator.num_customers > 0
    
    def test_generate_products(self, generator):
        """Test product generation."""
        products = generator.generate_products()
        
        assert isinstance(products, pd.DataFrame)
        assert len(products) > 0
        assert 'ProductID' in products.columns
        assert 'ProductName' in products.columns
        assert 'Category' in products.columns
        assert 'Price' in products.columns
        assert products['Price'].min() >= 0
        assert products['Price'].max() <= 300
    
    def test_generate_stores(self, generator):
        """Test store generation."""
        stores = generator.generate_stores()
        
        assert isinstance(stores, pd.DataFrame)
        assert len(stores) > 0
        assert 'StoreID' in stores.columns
        assert 'StoreName' in stores.columns
        assert 'City' in stores.columns
        assert 'Province' in stores.columns
    
    def test_generate_customers(self, generator):
        """Test customer generation."""
        customers = generator.generate_customers()
        
        assert isinstance(customers, pd.DataFrame)
        assert len(customers) == generator.num_customers
        assert 'CustomerID' in customers.columns
        assert 'FirstName' in customers.columns
        assert 'LastName' in customers.columns
        assert 'Email' in customers.columns
        assert 'EnrollmentDate' in customers.columns
        
        # Check for unique customer IDs
        assert customers['CustomerID'].nunique() == len(customers)
    
    def test_generate_transactions(self, generator):
        """Test transaction generation."""
        generator.generate_products()
        generator.generate_stores()
        generator.generate_customers()
        transactions = generator.generate_transactions()
        
        assert isinstance(transactions, pd.DataFrame)
        assert len(transactions) > 0
        assert 'TransactionID' in transactions.columns
        assert 'CustomerID' in transactions.columns
        assert 'TransactionDate' in transactions.columns
        assert 'StoreID' in transactions.columns
        assert 'ProductID' in transactions.columns
        assert 'Quantity' in transactions.columns
        assert 'UnitPrice' in transactions.columns
        assert 'LineTotal' in transactions.columns
        
        # Validate relationships
        assert all(cid in generator.customers['CustomerID'].values 
                  for cid in transactions['CustomerID'].unique())
        assert all(sid in generator.stores['StoreID'].values 
                  for sid in transactions['StoreID'].unique())
        assert all(pid in generator.products['ProductID'].values 
                  for pid in transactions['ProductID'].unique())
        
        # Validate calculations
        assert all(
            abs(row['LineTotal'] - (row['UnitPrice'] * row['Quantity'])) < 0.01
            for _, row in transactions.iterrows()
        )
    
    def test_generate_loyalty_data(self, generator):
        """Test loyalty data generation."""
        generator.generate_products()
        generator.generate_stores()
        generator.generate_customers()
        generator.generate_transactions()
        loyalty = generator.generate_loyalty_data()
        
        assert isinstance(loyalty, pd.DataFrame)
        assert len(loyalty) > 0
        assert 'CustomerID' in loyalty.columns
        assert 'TransactionID' in loyalty.columns
        assert 'PointsEarned' in loyalty.columns
        assert 'PointsBalance' in loyalty.columns
        assert 'Tier' in loyalty.columns
        
        # Validate points are non-negative
        assert all(loyalty['PointsEarned'] >= 0)
        assert all(loyalty['PointsBalance'] >= 0)
        
        # Validate tiers
        valid_tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
        assert all(tier in valid_tiers for tier in loyalty['Tier'].unique())
    
    def test_generate_survey_data(self, generator):
        """Test survey data generation."""
        generator.generate_products()
        generator.generate_stores()
        generator.generate_customers()
        generator.generate_transactions()
        surveys = generator.generate_survey_data()
        
        assert isinstance(surveys, pd.DataFrame)
        assert len(surveys) > 0
        assert 'CustomerID' in surveys.columns
        assert 'SatisfactionScore' in surveys.columns
        assert 'NPSScore' in surveys.columns
        assert 'Feedback' in surveys.columns
        
        # Validate score ranges
        assert all(1 <= score <= 5 for score in surveys['SatisfactionScore'])
        assert all(0 <= score <= 10 for score in surveys['NPSScore'])
    
    def test_generate_all_datasets(self, generator, tmp_path):
        """Test complete dataset generation."""
        output_dir = str(tmp_path / "data" / "raw")
        datasets = generator.generate_all_datasets(output_dir)
        
        assert isinstance(datasets, dict)
        assert 'transactions' in datasets
        assert 'loyalty' in datasets
        assert 'surveys' in datasets
        assert 'products' in datasets
        assert 'stores' in datasets
        assert 'customers' in datasets
        
        # Check files were created
        assert Path(output_dir, 'transactions.csv').exists()
        assert Path(output_dir, 'loyalty_pc_optimum.csv').exists()
        assert Path(output_dir, 'surveys.csv').exists()
        assert Path(output_dir, 'products.csv').exists()
        assert Path(output_dir, 'stores.csv').exists()
        assert Path(output_dir, 'customers.csv').exists()
    
    def test_data_relationships(self, generator):
        """Test that data relationships are maintained."""
        generator.generate_products()
        generator.generate_stores()
        generator.generate_customers()
        generator.generate_transactions()
        generator.generate_loyalty_data()
        generator.generate_survey_data()
        
        # All loyalty records should have corresponding transactions
        loyalty_trans_ids = set(generator.loyalty_df['TransactionID'].unique())
        trans_ids = set(generator.transactions_df['TransactionID'].unique())
        assert loyalty_trans_ids.issubset(trans_ids)
        
        # All survey customers should exist in customers
        survey_cust_ids = set(generator.survey_df['CustomerID'].unique())
        cust_ids = set(generator.customers['CustomerID'].unique())
        assert survey_cust_ids.issubset(cust_ids)
        
        # All transaction customers should exist
        trans_cust_ids = set(generator.transactions_df['CustomerID'].unique())
        assert trans_cust_ids.issubset(cust_ids)

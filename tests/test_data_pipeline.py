"""
Unit tests for data pipeline modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import extract, transform, load


class TestDataExtraction:
    """Test data extraction functions."""
    
    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create sample data files for testing."""
        data_dir = tmp_path / "data" / "raw"
        data_dir.mkdir(parents=True)
        
        # Create sample transactions
        transactions = pd.DataFrame({
            'TransactionID': ['TXN_000001', 'TXN_000002'],
            'CustomerID': ['CUST_00001', 'CUST_00002'],
            'TransactionDate': pd.to_datetime(['2023-01-15', '2023-01-16']),
            'StoreID': ['STORE_001', 'STORE_002'],
            'ProductID': ['PROD_00001', 'PROD_00002'],
            'Quantity': [2, 1],
            'UnitPrice': [10.99, 25.50],
            'LineTotal': [21.98, 25.50]
        })
        transactions.to_csv(data_dir / 'transactions.csv', index=False)
        
        # Create sample loyalty data
        loyalty = pd.DataFrame({
            'LoyaltyRecordID': [1, 2],
            'CustomerID': ['CUST_00001', 'CUST_00002'],
            'TransactionID': ['TXN_000001', 'TXN_000002'],
            'TransactionDate': pd.to_datetime(['2023-01-15', '2023-01-16']),
            'PointsEarned': [22, 26],
            'PointsRedeemed': [0, 0],
            'PointsBalance': [22, 26],
            'Tier': ['Bronze', 'Bronze'],
            'PointsPerDollar': [1.0, 1.0]
        })
        loyalty.to_csv(data_dir / 'loyalty_pc_optimum.csv', index=False)
        
        # Create sample surveys
        surveys = pd.DataFrame({
            'SurveyID': [1, 2],
            'CustomerID': ['CUST_00001', 'CUST_00002'],
            'SurveyDate': pd.to_datetime(['2023-02-01', '2023-02-02']),
            'SatisfactionScore': [4, 5],
            'NPSScore': [8, 9],
            'Feedback': ['Good', 'Excellent'],
            'WouldRecommend': ['Yes', 'Yes']
        })
        surveys.to_csv(data_dir / 'surveys.csv', index=False)
        
        # Create sample products
        products = pd.DataFrame({
            'ProductID': ['PROD_00001', 'PROD_00002'],
            'ProductName': ['Product 1', 'Product 2'],
            'Category': ['Electronics', 'Clothing'],
            'Price': [10.99, 25.50],
            'Cost': [5.00, 12.00]
        })
        products.to_csv(data_dir / 'products.csv', index=False)
        
        # Create sample stores
        stores = pd.DataFrame({
            'StoreID': ['STORE_001', 'STORE_002'],
            'StoreName': ['Store 1', 'Store 2'],
            'City': ['Toronto', 'Vancouver'],
            'Province': ['ON', 'BC'],
            'Address': ['123 Main St', '456 Oak Ave']
        })
        stores.to_csv(data_dir / 'stores.csv', index=False)
        
        # Create sample customers
        customers = pd.DataFrame({
            'CustomerID': ['CUST_00001', 'CUST_00002'],
            'FirstName': ['John', 'Jane'],
            'LastName': ['Doe', 'Smith'],
            'Email': ['john@example.com', 'jane@example.com'],
            'Phone': ['123-456-7890', '987-654-3210'],
            'Address': ['123 St', '456 Ave'],
            'City': ['Toronto', 'Vancouver'],
            'Province': ['ON', 'BC'],
            'PostalCode': ['M1A1A1', 'V1A1A1'],
            'EnrollmentDate': pd.to_datetime(['2022-01-01', '2022-02-01']),
            'DateOfBirth': pd.to_datetime(['1990-01-01', '1991-02-01'])
        })
        customers.to_csv(data_dir / 'customers.csv', index=False)
        
        return str(data_dir)
    
    def test_load_transactions(self, sample_data_dir):
        """Test loading transactions."""
        df = extract.load_transactions(f"{sample_data_dir}/transactions.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'TransactionID' in df.columns
        assert 'CustomerID' in df.columns
    
    def test_load_loyalty_data(self, sample_data_dir):
        """Test loading loyalty data."""
        df = extract.load_loyalty_data(f"{sample_data_dir}/loyalty_pc_optimum.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'CustomerID' in df.columns
        assert 'PointsBalance' in df.columns
    
    def test_load_survey_data(self, sample_data_dir):
        """Test loading survey data."""
        df = extract.load_survey_data(f"{sample_data_dir}/surveys.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'SatisfactionScore' in df.columns
    
    def test_load_products(self, sample_data_dir):
        """Test loading products."""
        df = extract.load_products(f"{sample_data_dir}/products.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'ProductID' in df.columns
    
    def test_load_stores(self, sample_data_dir):
        """Test loading stores."""
        df = extract.load_stores(f"{sample_data_dir}/stores.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'StoreID' in df.columns
    
    def test_load_customers(self, sample_data_dir):
        """Test loading customers."""
        df = extract.load_customers(f"{sample_data_dir}/customers.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'CustomerID' in df.columns
    
    def test_load_all_data(self, sample_data_dir):
        """Test loading all data."""
        datasets = extract.load_all_data(sample_data_dir)
        assert isinstance(datasets, dict)
        assert 'transactions' in datasets
        assert 'loyalty' in datasets
        assert 'surveys' in datasets
        assert 'products' in datasets
        assert 'stores' in datasets
        assert 'customers' in datasets
    
    def test_validate_data_sources(self, sample_data_dir):
        """Test data source validation."""
        results = extract.validate_data_sources(sample_data_dir)
        assert all(results.values())
    
    def test_load_transactions_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            extract.load_transactions("nonexistent_file.csv")
    
    def test_load_transactions_missing_columns(self, tmp_path):
        """Test error handling for missing columns."""
        invalid_file = tmp_path / "invalid.csv"
        pd.DataFrame({'col1': [1, 2]}).to_csv(invalid_file, index=False)
        
        with pytest.raises(ValueError):
            extract.load_transactions(str(invalid_file))


class TestDataTransformation:
    """Test data transformation functions."""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data."""
        return pd.DataFrame({
            'TransactionID': ['TXN_001', 'TXN_002', 'TXN_003', 'TXN_004'],
            'CustomerID': ['CUST_001', 'CUST_002', 'CUST_001', 'CUST_003'],
            'TransactionDate': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-15', '2023-01-17']),
            'StoreID': ['STORE_001', 'STORE_002', 'STORE_001', 'STORE_003'],
            'ProductID': ['PROD_001', 'PROD_002', 'PROD_001', 'PROD_003'],
            'Quantity': [2, 1, 2, 3],
            'UnitPrice': [10.99, 25.50, 10.99, 5.99],
            'LineTotal': [21.98, 25.50, 21.98, 17.97]
        })
    
    @pytest.fixture
    def sample_loyalty(self):
        """Create sample loyalty data."""
        return pd.DataFrame({
            'LoyaltyRecordID': [1, 2, 3],
            'CustomerID': ['CUST_001', 'CUST_002', 'CUST_003'],
            'TransactionID': ['TXN_001', 'TXN_002', 'TXN_004'],
            'TransactionDate': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-17']),
            'PointsEarned': [22, 26, 18],
            'PointsRedeemed': [0, 0, 0],
            'PointsBalance': [22, 26, 18],
            'Tier': ['Bronze', 'Silver', 'Bronze'],
            'PointsPerDollar': [1.0, 1.15, 1.0]
        })
    
    @pytest.fixture
    def sample_surveys(self):
        """Create sample survey data."""
        return pd.DataFrame({
            'SurveyID': [1, 2],
            'CustomerID': ['CUST_001', 'CUST_002'],
            'SurveyDate': pd.to_datetime(['2023-02-01', '2023-02-02']),
            'SatisfactionScore': [4, 5],
            'NPSScore': [8, 9],
            'Feedback': ['Good', 'Excellent'],
            'WouldRecommend': ['Yes', 'Yes']
        })
    
    def test_clean_transactions(self, sample_transactions):
        """Test transaction cleaning."""
        cleaned = transform.clean_transactions(sample_transactions)
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) <= len(sample_transactions)
        assert 'TransactionID' in cleaned.columns
    
    def test_clean_transactions_removes_duplicates(self, sample_transactions):
        """Test that duplicates are removed."""
        # Add duplicate
        duplicated = pd.concat([sample_transactions, sample_transactions.iloc[[0]]])
        cleaned = transform.clean_transactions(duplicated, remove_duplicates=True)
        assert len(cleaned) == len(sample_transactions)
    
    def test_clean_transactions_handles_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'TransactionID': ['TXN_001', 'TXN_002'],
            'CustomerID': ['CUST_001', None],
            'TransactionDate': pd.to_datetime(['2023-01-15', '2023-01-16']),
            'StoreID': ['STORE_001', 'STORE_002'],
            'ProductID': ['PROD_001', 'PROD_002'],
            'Quantity': [2, 1],
            'UnitPrice': [10.99, 25.50],
            'LineTotal': [21.98, 25.50]
        })
        cleaned = transform.clean_transactions(df)
        assert len(cleaned) == 1  # One row should be removed
    
    def test_clean_transactions_fixes_invalid_quantities(self):
        """Test fixing invalid quantities."""
        df = pd.DataFrame({
            'TransactionID': ['TXN_001'],
            'CustomerID': ['CUST_001'],
            'TransactionDate': pd.to_datetime(['2023-01-15']),
            'StoreID': ['STORE_001'],
            'ProductID': ['PROD_001'],
            'Quantity': [-1],  # Invalid
            'UnitPrice': [10.99],
            'LineTotal': [-10.99]
        })
        cleaned = transform.clean_transactions(df)
        assert cleaned['Quantity'].iloc[0] == 1  # Should be fixed to 1
    
    def test_clean_loyalty_data(self, sample_loyalty):
        """Test loyalty data cleaning."""
        cleaned = transform.clean_loyalty_data(sample_loyalty)
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) <= len(sample_loyalty)
    
    def test_clean_loyalty_data_validates_points(self):
        """Test point validation."""
        df = pd.DataFrame({
            'LoyaltyRecordID': [1],
            'CustomerID': ['CUST_001'],
            'TransactionID': ['TXN_001'],
            'TransactionDate': pd.to_datetime(['2023-01-15']),
            'PointsEarned': [-10],  # Invalid
            'PointsRedeemed': [0],
            'PointsBalance': [-10],  # Invalid
            'Tier': ['Bronze'],
            'PointsPerDollar': [1.0]
        })
        cleaned = transform.clean_loyalty_data(df)
        assert cleaned['PointsEarned'].iloc[0] >= 0
        assert cleaned['PointsBalance'].iloc[0] >= 0
    
    def test_clean_survey_data(self, sample_surveys):
        """Test survey data cleaning."""
        cleaned = transform.clean_survey_data(sample_surveys)
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) <= len(sample_surveys)
    
    def test_clean_survey_data_validates_scores(self):
        """Test score validation."""
        df = pd.DataFrame({
            'SurveyID': [1],
            'CustomerID': ['CUST_001'],
            'SurveyDate': pd.to_datetime(['2023-02-01']),
            'SatisfactionScore': [10],  # Invalid (should be 1-5)
            'NPSScore': [15],  # Invalid (should be 0-10)
            'Feedback': ['Test'],
            'WouldRecommend': ['Yes']
        })
        cleaned = transform.clean_survey_data(df)
        assert 1 <= cleaned['SatisfactionScore'].iloc[0] <= 5
        assert 0 <= cleaned['NPSScore'].iloc[0] <= 10
    
    def test_normalize_dates(self):
        """Test date normalization."""
        df = pd.DataFrame({
            'date1': ['2023-01-15', '2023-01-16'],
            'date2': ['2023-02-01', '2023-02-02']
        })
        normalized = transform.normalize_dates(df, ['date1', 'date2'])
        assert pd.api.types.is_datetime64_any_dtype(normalized['date1'])
        assert pd.api.types.is_datetime64_any_dtype(normalized['date2'])
    
    def test_validate_referential_integrity(self, sample_transactions, sample_loyalty, 
                                          sample_surveys):
        """Test referential integrity validation."""
        customers = pd.DataFrame({
            'CustomerID': ['CUST_001', 'CUST_002', 'CUST_003']
        })
        products = pd.DataFrame({
            'ProductID': ['PROD_001', 'PROD_002', 'PROD_003']
        })
        stores = pd.DataFrame({
            'StoreID': ['STORE_001', 'STORE_002', 'STORE_003']
        })
        
        results = transform.validate_referential_integrity(
            sample_transactions, sample_loyalty, sample_surveys,
            customers, products, stores
        )
        assert isinstance(results, dict)
        assert all(results.values())  # All should be valid
    
    def test_merge_datasets(self, sample_transactions, sample_loyalty, sample_surveys):
        """Test dataset merging."""
        customers = pd.DataFrame({
            'CustomerID': ['CUST_001', 'CUST_002', 'CUST_003'],
            'FirstName': ['John', 'Jane', 'Bob'],
            'LastName': ['Doe', 'Smith', 'Jones'],
            'EnrollmentDate': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01']),
            'City': ['Toronto', 'Vancouver', 'Montreal'],
            'Province': ['ON', 'BC', 'QC']
        })
        products = pd.DataFrame({
            'ProductID': ['PROD_001', 'PROD_002', 'PROD_003'],
            'ProductName': ['Product 1', 'Product 2', 'Product 3'],
            'Category': ['Electronics', 'Clothing', 'Food'],
            'Price': [10.99, 25.50, 5.99],
            'Cost': [5.00, 12.00, 2.00]
        })
        stores = pd.DataFrame({
            'StoreID': ['STORE_001', 'STORE_002', 'STORE_003'],
            'StoreName': ['Store 1', 'Store 2', 'Store 3'],
            'City': ['Toronto', 'Vancouver', 'Montreal'],
            'Province': ['ON', 'BC', 'QC']
        })
        
        merged = transform.merge_datasets(
            sample_transactions, sample_loyalty, sample_surveys,
            customers, products, stores
        )
        assert isinstance(merged, pd.DataFrame)
        assert len(merged) > 0
        assert 'ProductName' in merged.columns
        assert 'StoreName' in merged.columns
    
    def test_calculate_data_quality_metrics(self, sample_transactions):
        """Test data quality metrics calculation."""
        metrics = transform.calculate_data_quality_metrics(sample_transactions, "test")
        assert isinstance(metrics, dict)
        assert 'total_records' in metrics
        assert 'missing_values' in metrics
        assert 'duplicate_records' in metrics


class TestDataLoading:
    """Test data loading functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10.5, 20.3, 30.1]
        })
    
    def test_save_processed_data_parquet(self, sample_df, tmp_path):
        """Test saving to Parquet format."""
        file_path = tmp_path / "test_data"
        load.save_processed_data(sample_df, str(file_path), format='parquet')
        assert (tmp_path / "test_data.parquet").exists()
    
    def test_save_processed_data_csv(self, sample_df, tmp_path):
        """Test saving to CSV format."""
        file_path = tmp_path / "test_data"
        load.save_processed_data(sample_df, str(file_path), format='csv')
        assert (tmp_path / "test_data.csv").exists()
    
    def test_save_processed_data_both(self, sample_df, tmp_path):
        """Test saving to both formats."""
        file_path = tmp_path / "test_data"
        load.save_processed_data(sample_df, str(file_path), format='both')
        assert (tmp_path / "test_data.parquet").exists()
        assert (tmp_path / "test_data.csv").exists()
    
    def test_save_all_processed_data(self, tmp_path):
        """Test saving multiple datasets."""
        datasets = {
            'dataset1': pd.DataFrame({'col1': [1, 2]}),
            'dataset2': pd.DataFrame({'col2': [3, 4]})
        }
        output_dir = tmp_path / "output"
        load.save_all_processed_data(datasets, str(output_dir), format='parquet')
        assert (output_dir / "dataset1.parquet").exists()
        assert (output_dir / "dataset2.parquet").exists()
    
    def test_create_data_warehouse_structure(self, tmp_path):
        """Test creating data warehouse structure."""
        output_dir = tmp_path / "warehouse"
        load.create_data_warehouse_structure(str(output_dir))
        assert (output_dir / "fact_tables").exists()
        assert (output_dir / "dimension_tables").exists()
        assert (output_dir / "aggregated_tables").exists()
    
    def test_create_date_dimension(self, tmp_path):
        """Test creating date dimension."""
        dim_dates = load.create_date_dimension(
            start_date='2023-01-01',
            end_date='2023-12-31',
            output_dir=str(tmp_path)
        )
        assert isinstance(dim_dates, pd.DataFrame)
        assert len(dim_dates) == 365
        assert 'DateKey' in dim_dates.columns
        assert 'Year' in dim_dates.columns
        assert 'Month' in dim_dates.columns

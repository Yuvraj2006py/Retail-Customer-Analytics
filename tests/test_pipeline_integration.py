"""
Integration tests for the complete ETL pipeline.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import pipeline, extract, transform


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def sample_data_structure(self, tmp_path):
        """Create a complete sample data structure."""
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        
        # Create minimal but complete datasets
        transactions = pd.DataFrame({
            'TransactionID': ['TXN_000001', 'TXN_000002', 'TXN_000003'],
            'CustomerID': ['CUST_00001', 'CUST_00002', 'CUST_00001'],
            'TransactionDate': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-17']),
            'StoreID': ['STORE_001', 'STORE_002', 'STORE_001'],
            'ProductID': ['PROD_00001', 'PROD_00002', 'PROD_00001'],
            'Quantity': [2, 1, 3],
            'UnitPrice': [10.99, 25.50, 10.99],
            'LineTotal': [21.98, 25.50, 32.97]
        })
        transactions.to_csv(raw_dir / 'transactions.csv', index=False)
        
        loyalty = pd.DataFrame({
            'LoyaltyRecordID': [1, 2, 3],
            'CustomerID': ['CUST_00001', 'CUST_00002', 'CUST_00001'],
            'TransactionID': ['TXN_000001', 'TXN_000002', 'TXN_000003'],
            'TransactionDate': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-17']),
            'PointsEarned': [22, 26, 33],
            'PointsRedeemed': [0, 0, 0],
            'PointsBalance': [22, 26, 55],
            'Tier': ['Bronze', 'Bronze', 'Bronze'],
            'PointsPerDollar': [1.0, 1.0, 1.0]
        })
        loyalty.to_csv(raw_dir / 'loyalty_pc_optimum.csv', index=False)
        
        surveys = pd.DataFrame({
            'SurveyID': [1, 2],
            'CustomerID': ['CUST_00001', 'CUST_00002'],
            'SurveyDate': pd.to_datetime(['2023-02-01', '2023-02-02']),
            'SatisfactionScore': [4, 5],
            'NPSScore': [8, 9],
            'Feedback': ['Good', 'Excellent'],
            'WouldRecommend': ['Yes', 'Yes']
        })
        surveys.to_csv(raw_dir / 'surveys.csv', index=False)
        
        products = pd.DataFrame({
            'ProductID': ['PROD_00001', 'PROD_00002'],
            'ProductName': ['Product 1', 'Product 2'],
            'Category': ['Electronics', 'Clothing'],
            'Price': [10.99, 25.50],
            'Cost': [5.00, 12.00]
        })
        products.to_csv(raw_dir / 'products.csv', index=False)
        
        stores = pd.DataFrame({
            'StoreID': ['STORE_001', 'STORE_002'],
            'StoreName': ['Store 1', 'Store 2'],
            'City': ['Toronto', 'Vancouver'],
            'Province': ['ON', 'BC'],
            'Address': ['123 Main St', '456 Oak Ave']
        })
        stores.to_csv(raw_dir / 'stores.csv', index=False)
        
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
        customers.to_csv(raw_dir / 'customers.csv', index=False)
        
        return str(tmp_path)
    
    def test_full_pipeline(self, sample_data_structure):
        """Test the complete ETL pipeline."""
        raw_dir = Path(sample_data_structure) / "data" / "raw"
        processed_dir = Path(sample_data_structure) / "data" / "processed"
        warehouse_dir = Path(sample_data_structure) / "data" / "warehouse"
        
        # Run pipeline
        datasets = pipeline.run_etl_pipeline(
            raw_data_dir=str(raw_dir),
            processed_data_dir=str(processed_dir),
            warehouse_dir=str(warehouse_dir)
        )
        
        # Verify outputs
        assert isinstance(datasets, dict)
        assert 'transactions_cleaned' in datasets
        assert 'loyalty_cleaned' in datasets
        assert 'surveys_cleaned' in datasets
        assert 'merged_dataset' in datasets
        
        # Verify files were created
        assert (processed_dir / 'transactions_cleaned.parquet').exists()
        assert (processed_dir / 'loyalty_cleaned.parquet').exists()
        assert (processed_dir / 'surveys_cleaned.parquet').exists()
        assert (processed_dir / 'merged_dataset.parquet').exists()
        
        # Verify warehouse structure
        assert (warehouse_dir / 'fact_tables').exists()
        assert (warehouse_dir / 'dimension_tables').exists()
        assert (warehouse_dir / 'aggregated_tables').exists()
        
        # Verify fact tables
        assert (warehouse_dir / 'fact_tables' / 'fact_transactions.parquet').exists()
        assert (warehouse_dir / 'fact_tables' / 'fact_loyalty_points.parquet').exists()
        assert (warehouse_dir / 'fact_tables' / 'fact_surveys.parquet').exists()
        
        # Verify dimension tables
        assert (warehouse_dir / 'dimension_tables' / 'dim_customers.parquet').exists()
        assert (warehouse_dir / 'dimension_tables' / 'dim_products.parquet').exists()
        assert (warehouse_dir / 'dimension_tables' / 'dim_stores.parquet').exists()
        assert (warehouse_dir / 'dimension_tables' / 'dim_dates.parquet').exists()
        
        # Verify data integrity
        assert len(datasets['transactions_cleaned']) > 0
        assert len(datasets['merged_dataset']) > 0
        assert 'ProductName' in datasets['merged_dataset'].columns
        assert 'StoreName' in datasets['merged_dataset'].columns

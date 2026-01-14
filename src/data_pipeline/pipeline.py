"""
Main ETL Pipeline

This module orchestrates the complete ETL process:
1. Extract data from raw sources
2. Transform and clean data
3. Load processed data to files
"""

import logging
from pathlib import Path
from typing import Dict
import pandas as pd

from . import extract, transform, load

logger = logging.getLogger(__name__)


def run_etl_pipeline(raw_data_dir: str = "data/raw",
                     processed_data_dir: str = "data/processed",
                     warehouse_dir: str = "data/warehouse") -> Dict[str, pd.DataFrame]:
    """
    Run the complete ETL pipeline.
    
    Args:
        raw_data_dir: Directory containing raw data files
        processed_data_dir: Directory to save processed data
        warehouse_dir: Directory for data warehouse structure
        
    Returns:
        Dictionary with all processed DataFrames
    """
    logger.info("=" * 60)
    logger.info("Starting ETL Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Validate data sources
    logger.info("\nStep 1: Validating data sources...")
    extract.validate_data_sources(raw_data_dir)
    logger.info("✓ All data sources validated")
    
    # Step 2: Extract data
    logger.info("\nStep 2: Extracting data...")
    datasets = extract.load_all_data(raw_data_dir)
    logger.info(f"✓ Extracted {len(datasets)} datasets")
    
    # Step 3: Transform data
    logger.info("\nStep 3: Transforming and cleaning data...")
    
    # Clean individual datasets
    cleaned_transactions = transform.clean_transactions(datasets['transactions'])
    cleaned_loyalty = transform.clean_loyalty_data(datasets['loyalty'])
    cleaned_surveys = transform.clean_survey_data(datasets['surveys'])
    
    # Normalize dates
    cleaned_transactions = transform.normalize_dates(
        cleaned_transactions, ['TransactionDate']
    )
    cleaned_loyalty = transform.normalize_dates(
        cleaned_loyalty, ['TransactionDate']
    )
    cleaned_surveys = transform.normalize_dates(
        cleaned_surveys, ['SurveyDate']
    )
    
    logger.info("✓ Data cleaned and normalized")
    
    # Step 4: Validate referential integrity
    logger.info("\nStep 4: Validating referential integrity...")
    integrity_results = transform.validate_referential_integrity(
        cleaned_transactions,
        cleaned_loyalty,
        cleaned_surveys,
        datasets['customers'],
        datasets['products'],
        datasets['stores']
    )
    
    if all(integrity_results.values()):
        logger.info("✓ All referential integrity checks passed")
    else:
        logger.warning("⚠ Some referential integrity issues found (see details above)")
    
    # Step 5: Merge datasets
    logger.info("\nStep 5: Merging datasets...")
    merged_dataset = transform.merge_datasets(
        cleaned_transactions,
        cleaned_loyalty,
        cleaned_surveys,
        datasets['customers'],
        datasets['products'],
        datasets['stores']
    )
    logger.info(f"✓ Merged dataset created with {len(merged_dataset):,} records")
    
    # Step 6: Calculate data quality metrics
    logger.info("\nStep 6: Calculating data quality metrics...")
    transform.calculate_data_quality_metrics(cleaned_transactions, "Transactions")
    transform.calculate_data_quality_metrics(cleaned_loyalty, "Loyalty")
    transform.calculate_data_quality_metrics(cleaned_surveys, "Surveys")
    transform.calculate_data_quality_metrics(merged_dataset, "Merged Dataset")
    
    # Step 7: Prepare processed datasets
    processed_datasets = {
        'transactions_cleaned': cleaned_transactions,
        'loyalty_cleaned': cleaned_loyalty,
        'surveys_cleaned': cleaned_surveys,
        'merged_dataset': merged_dataset,
        'customers': datasets['customers'],
        'products': datasets['products'],
        'stores': datasets['stores']
    }
    
    # Step 8: Load processed data
    logger.info("\nStep 8: Loading processed data...")
    load.save_all_processed_data(processed_datasets, processed_data_dir, format='parquet')
    logger.info("✓ Processed data saved")
    
    # Step 9: Create data warehouse structure
    logger.info("\nStep 9: Creating data warehouse structure...")
    load.create_data_warehouse_structure(warehouse_dir)
    
    # Prepare fact and dimension tables
    load.prepare_fact_tables(
        cleaned_transactions,
        cleaned_loyalty,
        cleaned_surveys,
        warehouse_dir
    )
    
    load.prepare_dimension_tables(
        datasets['customers'],
        datasets['products'],
        datasets['stores'],
        warehouse_dir
    )
    
    # Create date dimension
    load.create_date_dimension(
        start_date='2020-01-01',
        end_date='2025-12-31',
        output_dir=warehouse_dir
    )
    
    logger.info("✓ Data warehouse structure created")
    
    logger.info("\n" + "=" * 60)
    logger.info("ETL Pipeline Completed Successfully!")
    logger.info("=" * 60)
    logger.info(f"\nProcessed datasets saved to: {processed_data_dir}")
    logger.info(f"Data warehouse created at: {warehouse_dir}")
    
    return processed_datasets


if __name__ == "__main__":
    # Run the pipeline
    datasets = run_etl_pipeline()
    print(f"\nPipeline completed! Processed {len(datasets)} datasets.")

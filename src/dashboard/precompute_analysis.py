"""
Pre-compute all analysis results for the dashboard.

This script generates all metrics, models, and visualizations data
so the dashboard can load them quickly without recomputing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import logging
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def precompute_all_analysis():
    """Pre-compute all analysis and save results."""
    
    logger.info("=" * 60)
    logger.info("Pre-computing Analysis Results")
    logger.info("=" * 60)
    
    output_dir = Path("data/dashboard_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import modules
        from src.data_pipeline import extract
        from src.features import feature_pipeline
        from src.models import segmentation, churn_prediction
        
        data_path = Path("data/processed")
        
        # Step 1: Load raw data
        logger.info("\nStep 1: Loading raw data...")
        datasets = {
            'transactions': extract.load_transactions(str(data_path / 'transactions_cleaned.parquet')),
            'loyalty': extract.load_loyalty_data(str(data_path / 'loyalty_cleaned.parquet')),
            'surveys': extract.load_survey_data(str(data_path / 'surveys_cleaned.parquet')),
            'customers': extract.load_customers(str(data_path / 'customers.parquet')),
            'products': extract.load_products(str(data_path / 'products.parquet')),
            'stores': extract.load_stores(str(data_path / 'stores.parquet'))
        }
        logger.info("Raw data loaded successfully")
        
        # Step 2: Create feature matrix
        logger.info("\nStep 2: Creating feature matrix...")
        feature_matrix = feature_pipeline.create_feature_matrix(
            transactions_df=datasets['transactions'],
            loyalty_df=datasets['loyalty'],
            customers_df=datasets['customers'],
            products_df=datasets['products'],
            surveys_df=datasets['surveys']
        )
        logger.info(f"Feature matrix created: {feature_matrix.shape}")
        
        # Save feature matrix as CSV (easier to load)
        feature_matrix.to_csv(output_dir / 'feature_matrix.csv', index=False)
        logger.info("Feature matrix saved")
        
        # Step 3: Perform segmentation
        logger.info("\nStep 3: Performing customer segmentation...")
        segmentation_results = segmentation.perform_customer_segmentation(
            feature_matrix,
            n_clusters=5,
            find_optimal_k=False,
            random_state=42
        )
        
        # Save segmentation results
        segmentation_results['labels'].to_csv(output_dir / 'segment_labels.csv')
        segmentation_results['profiles'].to_csv(output_dir / 'segment_profiles.csv', index=False)
        # Convert keys to regular ints for JSON serialization
        segment_names_clean = {int(k): str(v) for k, v in segmentation_results['segment_names'].items()}
        with open(output_dir / 'segment_names.json', 'w') as f:
            json.dump(segment_names_clean, f, indent=2)
        logger.info("Segmentation results saved")
        
        # Step 4: Perform churn prediction
        logger.info("\nStep 4: Performing churn prediction...")
        churn_results = churn_prediction.perform_churn_prediction(
            feature_matrix,
            datasets['transactions'],
            churn_threshold_days=90,
            use_smote=False,  # Use class weights instead for faster computation
            hyperparameters={
                'n_estimators': 50,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            random_state=42
        )
        
        # Save churn results
        churn_results['predictions'].to_csv(output_dir / 'churn_predictions.csv', index=False)
        churn_results['feature_importance'].to_csv(output_dir / 'churn_feature_importance.csv', index=False)
        
        # Save metrics
        metrics_to_save = {
            'roc_auc': float(churn_results['metrics']['roc_auc']),
            'f1_score': float(churn_results['metrics']['f1_score']),
            'precision': float(churn_results['metrics']['precision']),
            'recall': float(churn_results['metrics']['recall']),
            'accuracy': float(churn_results['metrics']['accuracy'])
        }
        with open(output_dir / 'churn_metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info("Churn prediction results saved")
        
        # Step 5: Compute dashboard metrics
        logger.info("\nStep 5: Computing dashboard metrics...")
        
        transactions = datasets['transactions']
        customers = datasets['customers']
        
        # KPI metrics
        kpi_metrics = {
            'total_customers': int(len(customers)),
            'total_revenue': float(transactions['LineTotal'].sum()),
            'avg_transaction_value': float(transactions['LineTotal'].mean()),
            'total_transactions': int(len(transactions)),
            'unique_products': int(transactions['ProductID'].nunique()),
            'active_customers': int((churn_results['predictions']['ChurnPrediction'] == 0).sum()),
            'churn_rate': float((churn_results['predictions']['ChurnPrediction'] == 1).sum() / len(churn_results['predictions']) * 100)
        }
        
        if 'LifetimeValue' in feature_matrix.columns:
            kpi_metrics['avg_lifetime_value'] = float(feature_matrix['LifetimeValue'].mean())
        else:
            kpi_metrics['avg_lifetime_value'] = 0.0
        
        with open(output_dir / 'kpi_metrics.json', 'w') as f:
            json.dump(kpi_metrics, f, indent=2)
        logger.info("KPI metrics saved")
        
        # Monthly revenue trend
        transactions['YearMonth'] = pd.to_datetime(transactions['TransactionDate']).dt.to_period('M').astype(str)
        monthly_revenue = transactions.groupby('YearMonth')['LineTotal'].sum().reset_index()
        monthly_revenue.to_csv(output_dir / 'monthly_revenue.csv', index=False)
        logger.info("Monthly revenue trend saved")
        
        # Product performance
        trans_with_products = transactions.merge(
            datasets['products'][['ProductID', 'ProductName', 'Category', 'Price']],
            on='ProductID',
            how='left'
        )
        
        product_revenue = trans_with_products.groupby(['ProductID', 'ProductName', 'Category']).agg({
            'LineTotal': 'sum',
            'Quantity': 'sum',
            'TransactionID': 'count'
        }).reset_index()
        product_revenue.columns = ['ProductID', 'ProductName', 'Category', 'TotalRevenue', 'TotalQuantity', 'TransactionCount']
        product_revenue = product_revenue.sort_values('TotalRevenue', ascending=False)
        product_revenue.to_csv(output_dir / 'product_performance.csv', index=False)
        logger.info("Product performance saved")
        
        # Category performance
        category_perf = trans_with_products.groupby('Category').agg({
            'LineTotal': 'sum',
            'Quantity': 'sum',
            'TransactionID': 'count'
        }).reset_index()
        category_perf.columns = ['Category', 'TotalRevenue', 'TotalQuantity', 'TransactionCount']
        category_perf = category_perf.sort_values('TotalRevenue', ascending=False)
        category_perf.to_csv(output_dir / 'category_performance.csv', index=False)
        logger.info("Category performance saved")
        
        # Segment distribution
        segment_counts = segmentation_results['labels'].value_counts().sort_index()
        segment_dist = pd.DataFrame({
            'Segment': [segmentation_results['segment_names'].get(i, f"Segment {i}") for i in segment_counts.index],
            'Count': segment_counts.values,
            'Percentage': (segment_counts.values / len(segmentation_results['labels']) * 100).round(2)
        })
        segment_dist.to_csv(output_dir / 'segment_distribution.csv', index=False)
        logger.info("Segment distribution saved")
        
        # Churn risk distribution
        risk_dist = churn_results['predictions']['RiskLevel'].value_counts()
        risk_dist_df = pd.DataFrame({
            'RiskLevel': risk_dist.index,
            'Count': risk_dist.values,
            'Percentage': (risk_dist.values / len(churn_results['predictions']) * 100).round(2)
        })
        risk_dist_df.to_csv(output_dir / 'churn_risk_distribution.csv', index=False)
        logger.info("Churn risk distribution saved")
        
        # Save timestamp
        with open(output_dir / 'last_updated.txt', 'w') as f:
            f.write(datetime.now().isoformat())
        
        logger.info("\n" + "=" * 60)
        logger.info("Pre-computation Complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during pre-computation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = precompute_all_analysis()
    if success:
        print("\n[SUCCESS] All analysis pre-computed successfully!")
        print("You can now run the dashboard and it will load instantly.")
    else:
        print("\n[ERROR] Pre-computation failed. Check the logs above.")

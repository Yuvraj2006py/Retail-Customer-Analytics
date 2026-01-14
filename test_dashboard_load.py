"""Test script to verify dashboard data loading."""

from pathlib import Path
import pandas as pd
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_load_precomputed_data():
    """Test loading pre-computed data."""
    cache_dir = Path("data/dashboard_cache")
    
    print(f"Cache directory exists: {cache_dir.exists()}")
    print(f"Cache directory path: {cache_dir.absolute()}")
    
    if not cache_dir.exists():
        print("ERROR: Cache directory does not exist!")
        return False
    
    required_files = {
        'feature_matrix.csv': lambda: pd.read_csv(cache_dir / 'feature_matrix.csv'),
        'segment_labels.csv': lambda: pd.read_csv(cache_dir / 'segment_labels.csv', index_col=0),
        'segment_profiles.csv': lambda: pd.read_csv(cache_dir / 'segment_profiles.csv'),
        'segment_names.json': lambda: json.load(open(cache_dir / 'segment_names.json', 'r')),
        'churn_predictions.csv': lambda: pd.read_csv(cache_dir / 'churn_predictions.csv'),
        'churn_feature_importance.csv': lambda: pd.read_csv(cache_dir / 'churn_feature_importance.csv'),
        'churn_metrics.json': lambda: json.load(open(cache_dir / 'churn_metrics.json', 'r')),
        'kpi_metrics.json': lambda: json.load(open(cache_dir / 'kpi_metrics.json', 'r')),
        'monthly_revenue.csv': lambda: pd.read_csv(cache_dir / 'monthly_revenue.csv'),
        'category_performance.csv': lambda: pd.read_csv(cache_dir / 'category_performance.csv'),
        'segment_distribution.csv': lambda: pd.read_csv(cache_dir / 'segment_distribution.csv'),
        'churn_risk_distribution.csv': lambda: pd.read_csv(cache_dir / 'churn_risk_distribution.csv'),
    }
    
    print("\nTesting file loading...")
    all_ok = True
    
    for filename, load_func in required_files.items():
        filepath = cache_dir / filename
        if not filepath.exists():
            print(f"  [X] {filename}: FILE NOT FOUND")
            all_ok = False
        else:
            try:
                data = load_func()
                if isinstance(data, pd.DataFrame):
                    print(f"  [OK] {filename}: Loaded {len(data)} rows")
                elif isinstance(data, dict):
                    print(f"  [OK] {filename}: Loaded {len(data)} keys")
                else:
                    print(f"  [OK] {filename}: Loaded successfully")
            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")
                all_ok = False
    
    # Test product_performance (optional)
    product_perf_path = cache_dir / 'product_performance.csv'
    if product_perf_path.exists():
        try:
            product_perf = pd.read_csv(product_perf_path)
            print(f"  [OK] product_performance.csv: Loaded {len(product_perf)} rows (optional)")
        except Exception as e:
            print(f"  [WARN] product_performance.csv: {e} (optional file)")
    else:
        print(f"  [WARN] product_performance.csv: Not found (optional file)")
    
    if all_ok:
        print("\n[SUCCESS] All required files loaded successfully!")
        return True
    else:
        print("\n[FAILED] Some files failed to load!")
        return False

if __name__ == "__main__":
    success = test_load_precomputed_data()
    sys.exit(0 if success else 1)

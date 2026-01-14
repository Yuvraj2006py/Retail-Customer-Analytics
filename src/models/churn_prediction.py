"""
Churn Prediction Module

This module implements XGBoost-based churn prediction with:
- Temporal target variable creation
- Feature engineering for churn
- Hyperparameter tuning
- Class imbalance handling
- Model evaluation and interpretability
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import SMOTE, but make it optional
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logger.warning("imbalanced-learn not available. SMOTE will be disabled. Install with: pip install imbalanced-learn")


def create_churn_target(transactions_df: pd.DataFrame,
                       reference_date: Optional[pd.Timestamp] = None,
                       churn_threshold_days: int = 90) -> pd.DataFrame:
    """
    Create churn target variable based on recency.
    
    Churn = 1 if customer has no purchase in last churn_threshold_days
    Churn = 0 if customer has purchase in last churn_threshold_days
    
    Args:
        transactions_df: DataFrame with transaction data
        reference_date: Reference date for churn calculation (default: max transaction date)
        churn_threshold_days: Days without purchase to be considered churned
        
    Returns:
        DataFrame with CustomerID and Churn label
    """
    if reference_date is None:
        reference_date = transactions_df['TransactionDate'].max()
    
    logger.info(f"Creating churn target with threshold: {churn_threshold_days} days")
    logger.info(f"Reference date: {reference_date}")
    
    # Get last purchase date per customer
    last_purchase = transactions_df.groupby('CustomerID')['TransactionDate'].max().reset_index()
    last_purchase.columns = ['CustomerID', 'LastPurchaseDate']
    
    # Calculate days since last purchase
    last_purchase['DaysSinceLastPurchase'] = (
        reference_date - last_purchase['LastPurchaseDate']
    ).dt.days
    
    # Create churn label (1 = churned, 0 = active)
    last_purchase['Churn'] = (last_purchase['DaysSinceLastPurchase'] > churn_threshold_days).astype(int)
    
    churn_rate = last_purchase['Churn'].mean() * 100
    logger.info(f"Churn rate: {churn_rate:.2f}% ({last_purchase['Churn'].sum()} churned customers)")
    
    return last_purchase[['CustomerID', 'Churn', 'DaysSinceLastPurchase', 'LastPurchaseDate']]


def prepare_churn_features(feature_matrix: pd.DataFrame,
                          churn_target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for churn prediction.
    
    Args:
        feature_matrix: Customer feature matrix
        churn_target: DataFrame with churn labels
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    logger.info("Preparing features for churn prediction")
    
    # Merge features with churn target
    churn_data = feature_matrix.merge(churn_target[['CustomerID', 'Churn']], on='CustomerID', how='inner')
    
    # Separate features and target
    exclude_cols = ['CustomerID', 'Churn', 'LastPurchaseDate', 'FirstPurchaseDate']
    feature_cols = [col for col in churn_data.columns if col not in exclude_cols]
    
    # Select only numeric features
    numeric_features = churn_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = churn_data[numeric_features].copy()
    y = churn_data['Churn'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
    logger.info(f"Churn distribution: {y.value_counts().to_dict()}")
    
    return X, y


def create_temporal_split(feature_matrix: pd.DataFrame,
                         transactions_df: pd.DataFrame,
                         train_end_date: Optional[pd.Timestamp] = None,
                         churn_threshold_days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create temporal train/test split to avoid data leakage.
    
    Args:
        feature_matrix: Customer feature matrix
        transactions_df: Transaction data for temporal split
        train_end_date: End date for training data (default: 80% of date range)
        churn_threshold_days: Days threshold for churn
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Creating temporal train/test split")
    
    # Determine date range
    min_date = transactions_df['TransactionDate'].min()
    max_date = transactions_df['TransactionDate'].max()
    
    if train_end_date is None:
        # Use 80% of date range for training
        date_range = (max_date - min_date).days
        train_end_date = min_date + timedelta(days=int(date_range * 0.8))
    
    logger.info(f"Training period: {min_date.date()} to {train_end_date.date()}")
    logger.info(f"Test period: {train_end_date.date()} to {max_date.date()}")
    
    # Create churn target for training period
    train_transactions = transactions_df[transactions_df['TransactionDate'] <= train_end_date]
    train_churn = create_churn_target(train_transactions, reference_date=train_end_date, 
                                     churn_threshold_days=churn_threshold_days)
    
    # Create churn target for test period
    test_transactions = transactions_df[transactions_df['TransactionDate'] > train_end_date]
    test_churn = create_churn_target(test_transactions, reference_date=max_date,
                                     churn_threshold_days=churn_threshold_days)
    
    # Prepare features
    X_train, y_train = prepare_churn_features(feature_matrix, train_churn)
    X_test, y_test = prepare_churn_features(feature_matrix, test_churn)
    
    # Align indices - use CustomerID from feature_matrix if available
    # Otherwise, use common indices
    if 'CustomerID' in feature_matrix.columns:
        # Get customer IDs from original feature matrix
        train_customers = set(train_churn['CustomerID'].unique())
        test_customers = set(test_churn['CustomerID'].unique())
        common_customers = train_customers & test_customers
        
        if len(common_customers) > 0:
            if len(common_customers) < len(train_customers) or len(common_customers) < len(test_customers):
                logger.warning(f"Using {len(common_customers)} common customers for train/test split")
            # Filter to common customers
            train_churn_filtered = train_churn[train_churn['CustomerID'].isin(common_customers)]
            test_churn_filtered = test_churn[test_churn['CustomerID'].isin(common_customers)]
            X_train, y_train = prepare_churn_features(feature_matrix, train_churn_filtered)
            X_test, y_test = prepare_churn_features(feature_matrix, test_churn_filtered)
        else:
            logger.warning("No common customers found between train and test periods")
    else:
        # Use common indices
        common_indices = list(set(X_train.index) & set(X_test.index))
        if len(common_indices) > 0:
            if len(common_indices) < len(X_train) or len(common_indices) < len(X_test):
                logger.warning(f"Using {len(common_indices)} common indices for train/test split")
            X_train = X_train.loc[common_indices]
            y_train = y_train.loc[common_indices]
            X_test = X_test.loc[common_indices]
            y_test = y_test.loc[common_indices]
        else:
            logger.warning("No common indices found between train and test sets")
    
    logger.info(f"Train set: {len(X_train)} samples (churn rate: {y_train.mean()*100:.2f}%)")
    logger.info(f"Test set: {len(X_test)} samples (churn rate: {y_test.mean()*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test


def handle_class_imbalance(X: pd.DataFrame,
                          y: pd.Series,
                          method: str = 'smote',
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE or class weights.
    
    Args:
        X: Feature matrix
        y: Target labels
        method: 'smote' or 'none'
        random_state: Random state
        
    Returns:
        Tuple of (balanced X, balanced y)
    """
    logger.info(f"Handling class imbalance using method: {method}")
    logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
    
    if method == 'smote':
        if not SMOTE_AVAILABLE:
            logger.warning("SMOTE requested but imbalanced-learn not available. Using class weights instead.")
            X_balanced, y_balanced = X.copy(), y.copy()
        else:
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, y.sum() - 1) if y.sum() > 1 else 1)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Convert back to DataFrame
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns, index=range(len(X_balanced)))
            y_balanced = pd.Series(y_balanced, name='Churn')
            
            logger.info(f"After SMOTE: {len(X_balanced)} samples")
            logger.info(f"Balanced class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
    else:
        X_balanced, y_balanced = X.copy(), y.copy()
        logger.info("No balancing applied")
    
    return X_balanced, y_balanced


def train_xgboost_model(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None,
                        hyperparameters: Optional[Dict] = None,
                        use_class_weights: bool = True,
                        random_state: int = 42) -> xgb.XGBClassifier:
    """
    Train XGBoost churn prediction model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        hyperparameters: Model hyperparameters
        use_class_weights: Whether to use class weights for imbalance
        random_state: Random state
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost churn prediction model")
    
    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': random_state,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
    
    # Check if we have both classes
    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Need at least 2 classes for binary classification. Found: {unique_classes}")
    
    # Calculate class weights if needed
    if use_class_weights:
        class_counts = y_train.value_counts()
        # Handle case where we have both classes
        if len(class_counts) == 2 and 0 in class_counts.index and 1 in class_counts.index:
            scale_pos_weight = class_counts[0] / class_counts[1]
            hyperparameters['scale_pos_weight'] = scale_pos_weight
            logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
        else:
            logger.warning(f"Only one class present in training data: {class_counts.to_dict()}, skipping class weights")
    
    # Create model
    model = xgb.XGBClassifier(**hyperparameters)
    
    # Train with validation set if provided
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    
    logger.info("Model training complete")
    
    return model


def evaluate_model(model: xgb.XGBClassifier,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  threshold: float = 0.5) -> Dict:
    """
    Evaluate churn prediction model.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }
    
    logger.info(f"Model Performance:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


def calculate_feature_importance(model: xgb.XGBClassifier,
                                feature_names: List[str]) -> pd.DataFrame:
    """
    Calculate and rank feature importance.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    importance = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return feature_importance_df


def calculate_shap_values(model: xgb.XGBClassifier,
                          X_sample: pd.DataFrame,
                          max_samples: int = 1000) -> Optional[np.ndarray]:
    """
    Calculate SHAP values for model interpretability.
    
    Args:
        model: Trained XGBoost model
        X_sample: Sample features for SHAP calculation
        max_samples: Maximum samples to use (for performance)
        
    Returns:
        SHAP values array
    """
    try:
        import shap
        
        logger.info("Calculating SHAP values")
        
        # Sample data if too large
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(n=max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} instances for SHAP calculation")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        logger.info("SHAP values calculated successfully")
        return shap_values
        
    except ImportError:
        logger.warning("SHAP not available, skipping SHAP calculation")
        return None
    except Exception as e:
        logger.warning(f"Error calculating SHAP values: {e}")
        return None


def predict_churn_risk(model: xgb.XGBClassifier,
                      feature_matrix: pd.DataFrame,
                      threshold: float = 0.5) -> pd.DataFrame:
    """
    Predict churn risk for all customers.
    
    Args:
        model: Trained XGBoost model
        feature_matrix: Customer feature matrix
        threshold: Classification threshold
        
    Returns:
        DataFrame with churn predictions and probabilities
    """
    logger.info("Predicting churn risk for all customers")
    
    # Prepare features (same as training)
    exclude_cols = ['CustomerID', 'Churn', 'LastPurchaseDate', 'FirstPurchaseDate']
    feature_cols = [col for col in feature_matrix.columns if col not in exclude_cols]
    numeric_features = feature_matrix[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = feature_matrix[numeric_features].copy()
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Predict
    churn_proba = model.predict_proba(X)[:, 1]
    churn_pred = (churn_proba >= threshold).astype(int)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'CustomerID': feature_matrix['CustomerID'].values,
        'ChurnProbability': churn_proba,
        'ChurnPrediction': churn_pred,
        'RiskLevel': pd.cut(churn_proba, 
                           bins=[0, 0.3, 0.6, 0.8, 1.0],
                           labels=['Low', 'Medium', 'High', 'Critical'])
    })
    
    # Risk level distribution
    logger.info("Churn risk distribution:")
    for level in ['Low', 'Medium', 'High', 'Critical']:
        count = (results['RiskLevel'] == level).sum()
        pct = count / len(results) * 100
        logger.info(f"  {level}: {count} customers ({pct:.1f}%)")
    
    return results


def perform_churn_prediction(feature_matrix: pd.DataFrame,
                            transactions_df: pd.DataFrame,
                            churn_threshold_days: int = 90,
                            train_end_date: Optional[pd.Timestamp] = None,
                            use_smote: bool = True,
                            hyperparameters: Optional[Dict] = None,
                            random_state: int = 42) -> Dict:
    """
    Complete churn prediction pipeline.
    
    Args:
        feature_matrix: Customer feature matrix
        transactions_df: Transaction data
        churn_threshold_days: Days threshold for churn
        train_end_date: End date for training (None = 80% split)
        use_smote: Whether to use SMOTE for balancing
        hyperparameters: Model hyperparameters
        random_state: Random state
        
    Returns:
        Dictionary with model and results
    """
    logger.info("=" * 60)
    logger.info("Churn Prediction Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Create temporal split
    logger.info("\nStep 1: Creating temporal train/test split...")
    X_train, X_test, y_train, y_test = create_temporal_split(
        feature_matrix, transactions_df, train_end_date, churn_threshold_days
    )
    
    # Step 2: Handle class imbalance
    logger.info("\nStep 2: Handling class imbalance...")
    if use_smote:
        X_train_balanced, y_train_balanced = handle_class_imbalance(
            X_train, y_train, method='smote', random_state=random_state
        )
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Step 3: Train model
    logger.info("\nStep 3: Training XGBoost model...")
    model = train_xgboost_model(
        X_train_balanced, y_train_balanced,
        hyperparameters=hyperparameters,
        use_class_weights=not use_smote,  # Use weights if not using SMOTE
        random_state=random_state
    )
    
    # Step 4: Evaluate model
    logger.info("\nStep 4: Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 5: Feature importance
    logger.info("\nStep 5: Calculating feature importance...")
    feature_importance = calculate_feature_importance(model, X_train.columns.tolist())
    
    # Step 6: SHAP values (optional, can be slow)
    logger.info("\nStep 6: Calculating SHAP values...")
    shap_values = calculate_shap_values(model, X_test, max_samples=500)
    
    # Step 7: Predict on all customers
    logger.info("\nStep 7: Predicting churn risk for all customers...")
    churn_predictions = predict_churn_risk(model, feature_matrix)
    
    results = {
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'shap_values': shap_values,
        'predictions': churn_predictions,
        'X_train': X_train,
        'y_test': y_test,
        'test_predictions': model.predict_proba(X_test)[:, 1]
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("Churn Prediction Complete!")
    logger.info("=" * 60)
    logger.info(f"\nModel Performance:")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    
    return results


def save_churn_model(model: xgb.XGBClassifier,
                    results: Dict,
                    output_path: str) -> None:
    """
    Save churn prediction model and results.
    
    Args:
        model: Trained XGBoost model
        results: Results dictionary
        output_path: Output directory path
    """
    from pathlib import Path
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving churn model to {output_dir}")
    
    # Save model
    joblib.dump(model, output_dir / 'churn_model.pkl')
    
    # Save predictions
    results['predictions'].to_csv(output_dir / 'churn_predictions.csv', index=False)
    
    # Save feature importance
    results['feature_importance'].to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Save metrics
    import json
    metrics_save = {k: v for k, v in results['metrics'].items() if k != 'classification_report'}
    with open(output_dir / 'model_metrics.json', 'w') as f:
        json.dump(metrics_save, f, indent=2, default=str)
    
    logger.info("Churn model and results saved successfully")

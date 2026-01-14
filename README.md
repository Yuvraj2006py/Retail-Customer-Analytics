# Retail Customer Engagement & Loyalty Analytics

A comprehensive end-to-end analytics project for retail customer engagement and loyalty program analysis using SQL, Python (Pandas, NumPy, Scikit-learn, XGBoost), and Streamlit.

## Project Overview

This project analyzes transactional, loyalty (PC Optimum), and survey data to uncover consumer behavior patterns and provide actionable recommendations for marketing and inventory strategies. It includes:

- **Multi-source data pipeline** integrating transactional, loyalty, and survey datasets
- **Customer segmentation** using K-Means clustering
- **Churn prediction** using XGBoost
- **Interactive Streamlit dashboard** for visualization and insights
- **Comprehensive findings document** with visualizations

## Features

### Data Pipeline
- Synthetic data generation for testing and development
- ETL pipeline with extraction, transformation, and loading
- Data warehouse structure (star schema)
- Data quality validation and referential integrity checks

### Feature Engineering
- RFM (Recency, Frequency, Monetary) analysis
- Loyalty program metrics
- Engagement and behavioral features
- Seasonal and product performance features

### Machine Learning Models
- **Customer Segmentation**: K-Means clustering with optimal K selection
- **Churn Prediction**: XGBoost classifier with temporal train/test split
- Class imbalance handling (SMOTE/class weights)
- Model interpretability (SHAP values, feature importance)

### Dashboard
- Executive summary with key metrics
- Customer segment analysis
- Churn risk analysis
- Product performance insights
- Loyalty program metrics
- Findings & visualizations page

## Project Structure

```
Retail/
├── config/                 # Configuration files
├── data/
│   ├── raw/               # Raw data files
│   ├── processed/        # Processed data files
│   ├── dashboard_cache/  # Pre-computed analysis results
│   └── warehouse/        # Data warehouse tables
├── src/
│   ├── data_generation/  # Synthetic data generation
│   ├── data_pipeline/    # ETL pipeline
│   ├── features/         # Feature engineering
│   ├── models/          # ML models (segmentation, churn)
│   ├── dashboard/       # Streamlit dashboard
│   └── utils/           # Utility functions
├── sql/                  # SQL schemas and queries
├── tests/               # Unit and integration tests
├── visualizations/      # Generated visualization images
├── FINDINGS.md          # Comprehensive findings document
├── PROJECT_PLAN.md      # Detailed project plan
└── requirements.txt     # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Retail
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data

```bash
python src/data_generation/generate_synthetic_data.py
```

This will generate synthetic datasets in `data/raw/` including:
- Products, Stores, Customers
- Transactions
- Loyalty program data (PC Optimum)
- Survey data

### 2. Run Data Pipeline

```bash
python src/data_pipeline/pipeline.py
```

This will:
- Extract data from raw files
- Transform and clean the data
- Load into processed files and data warehouse

### 3. Pre-compute Analysis Results

```bash
python src/dashboard/precompute_analysis.py
```

This will generate:
- Feature matrix
- Customer segments
- Churn predictions
- Pre-computed metrics for dashboard

### 4. Generate Visualizations

```bash
python generate_findings_visualizations.py
```

This will create visualization images in `visualizations/` directory.

### 5. Run Dashboard

**Local Development:**
```bash
streamlit run src/dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

**Deploy to Streamlit Cloud:**
1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Sign in with GitHub
4. Click "New app"
5. Select repository: `Yuvraj2006py/Retail-Customer-Analytics`
6. Main file path: `src/dashboard/app.py`
7. Click "Deploy"

Your dashboard will be live at a URL like: `https://retail-customer-analytics-xxxxx.streamlit.app`

## Running Tests

```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_data_generation.py
pytest tests/test_data_pipeline.py
pytest tests/test_feature_engineering.py
pytest tests/test_segmentation.py
pytest tests/test_churn_prediction.py
```

## Key Findings

See [FINDINGS.md](FINDINGS.md) for comprehensive analysis including:
- Customer segmentation insights
- Churn prediction results
- Product performance analysis
- Loyalty program metrics
- Actionable recommendations

## Technologies Used

- **Python**: Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn, Plotly, Streamlit
- **Data Processing**: SQL, ETL pipelines
- **Testing**: Pytest
- **Model Interpretability**: SHAP

## Project Phases

1. [COMPLETE] Phase 0: Project setup and data generation
2. [COMPLETE] Phase 1: Data pipeline (ETL)
3. [COMPLETE] Phase 2: Feature engineering
4. [COMPLETE] Phase 3: Customer segmentation (K-Means)
5. [COMPLETE] Phase 4: Churn prediction (XGBoost)
6. [COMPLETE] Phase 5: Streamlit dashboard
7. [COMPLETE] Phase 6: Findings and visualizations

## Configuration

Edit configuration files in `config/`:
- `config.yaml`: Main project configuration
- `data_generation_config.yaml`: Synthetic data generation parameters

## Data Sources

The project uses synthetic data that mimics:
- Retail transactions
- PC Optimum loyalty program data
- Customer satisfaction surveys

For production use, replace synthetic data generation with real data sources.

## License

This project is for educational and portfolio purposes.

## Author

Retail Analytics Project

"""
Main Streamlit Dashboard Application

Retail Customer Engagement & Loyalty Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import json
import plotly.express as px

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure page
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1565a0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    h2 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #424242;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'loading' not in st.session_state:
    st.session_state.loading = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)
def load_precomputed_data():
    """Load pre-computed analysis results from cache."""
    cache_dir = Path("data/dashboard_cache")
    
    if not cache_dir.exists():
        return None
    
    try:
        logger.info("Loading pre-computed analysis results...")
        
        # Load feature matrix
        feature_matrix = pd.read_csv(cache_dir / 'feature_matrix.csv')
        
        # Load segmentation results
        segment_labels_df = pd.read_csv(cache_dir / 'segment_labels.csv', index_col=0)
        segment_labels = segment_labels_df.iloc[:, 0] if len(segment_labels_df.columns) > 0 else segment_labels_df.index
        segment_profiles = pd.read_csv(cache_dir / 'segment_profiles.csv')
        with open(cache_dir / 'segment_names.json', 'r') as f:
            segment_names = json.load(f)
        
        segmentation_results = {
            'labels': segment_labels,
            'profiles': segment_profiles,
            'segment_names': {int(k): v for k, v in segment_names.items()},
            'n_clusters': len(segment_profiles)
        }
        
        # Load churn results
        churn_predictions = pd.read_csv(cache_dir / 'churn_predictions.csv')
        churn_feature_importance = pd.read_csv(cache_dir / 'churn_feature_importance.csv')
        with open(cache_dir / 'churn_metrics.json', 'r') as f:
            churn_metrics = json.load(f)
        
        churn_results = {
            'predictions': churn_predictions,
            'feature_importance': churn_feature_importance,
            'metrics': churn_metrics
        }
        
        # Load KPI metrics
        with open(cache_dir / 'kpi_metrics.json', 'r') as f:
            kpi_metrics = json.load(f)
        
        # Load trend data
        monthly_revenue = pd.read_csv(cache_dir / 'monthly_revenue.csv')
        
        # Load product performance (handle missing file gracefully)
        product_performance_path = cache_dir / 'product_performance.csv'
        if product_performance_path.exists():
            product_performance = pd.read_csv(product_performance_path)
        else:
            logger.warning("product_performance.csv not found, creating empty DataFrame")
            product_performance = pd.DataFrame(columns=['ProductID', 'ProductName', 'Category', 'TotalRevenue', 'TotalQuantity', 'TransactionCount'])
        
        category_performance = pd.read_csv(cache_dir / 'category_performance.csv')
        segment_distribution = pd.read_csv(cache_dir / 'segment_distribution.csv')
        churn_risk_distribution = pd.read_csv(cache_dir / 'churn_risk_distribution.csv')
        
        # Combine all data
        all_data = {
            'feature_matrix': feature_matrix,
            'segmentation': segmentation_results,
            'churn': churn_results,
            'kpi_metrics': kpi_metrics,
            'monthly_revenue': monthly_revenue,
            'product_performance': product_performance,
            'category_performance': category_performance,
            'segment_distribution': segment_distribution,
            'churn_risk_distribution': churn_risk_distribution
        }
        
        logger.info("Pre-computed data loaded successfully")
        return all_data
        
    except Exception as e:
        logger.error(f"Error loading pre-computed data: {e}")
        return None


def main():
    """Main dashboard application."""
    
    # Auto-load pre-computed data on first run
    if not st.session_state.data_loaded and not st.session_state.loading:
        st.session_state.loading = True
        with st.spinner("Loading pre-computed analysis results..."):
            data = load_precomputed_data()
            if data:
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.session_state.loading = False
                st.rerun()
            else:
                st.session_state.loading = False
                st.error("""
                **Pre-computed data not found!**
                
                Please run the pre-computation script first:
                ```
                python src/dashboard/precompute_analysis.py
                ```
                
                This will generate all analysis results that the dashboard needs.
                """)
                return
    
    # Show loading state
    if st.session_state.loading:
        st.markdown('<h1 class="main-header">Retail Customer Engagement & Loyalty Analytics</h1>', 
                    unsafe_allow_html=True)
        st.info("Loading data and generating insights... Please wait.")
        return
    
    # Check if data failed to load
    if not st.session_state.data_loaded:
        st.markdown('<h1 class="main-header">Retail Customer Engagement & Loyalty Analytics</h1>', 
                    unsafe_allow_html=True)
        st.error("Failed to load data. Please check that all data files exist in data/processed/")
        if st.button("Retry Loading Data"):
            st.session_state.loading = True
            st.rerun()
        return
    
    # Header
    st.markdown('<h1 class="main-header">Retail Customer Engagement & Loyalty Analytics</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Retail+Analytics", 
                use_container_width=True)
        st.markdown("### Navigation")
        
        page = st.radio(
            "Select a page:",
            ["Executive Summary", 
             "Customer Segments", 
             "Churn Risk Analysis",
             "Product Performance",
             "Loyalty Program",
             "Findings & Visualizations"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Data loading status
        if st.session_state.data_loaded:
            st.success("Data loaded successfully")
        
        if st.button("Refresh Data", use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.data = {}
            st.session_state.loading = True
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Info")
        cache_dir = Path("data/dashboard_cache")
        if cache_dir.exists() and (cache_dir / 'last_updated.txt').exists():
            with open(cache_dir / 'last_updated.txt', 'r') as f:
                last_updated = f.read().strip()
            st.caption(f"Last updated: {last_updated}")
    
    data = st.session_state.data
    
    # Route to appropriate page
    if page == "Executive Summary":
        show_executive_summary(data)
    elif page == "Customer Segments":
        show_customer_segments(data)
    elif page == "Churn Risk Analysis":
        show_churn_risk(data)
    elif page == "Product Performance":
        show_product_performance(data)
    elif page == "Loyalty Program":
        show_loyalty_program(data)
    elif page == "Findings & Visualizations":
        show_findings_visualizations(data)


def show_executive_summary(data):
    """Display executive summary page."""
    st.header("Executive Summary")
    st.markdown("**Overview of key metrics and insights from your customer data**")
    st.markdown("---")
    st.markdown("### Key Performance Indicators")
    
    # Load KPIs from pre-computed data
    kpi_metrics = data['kpi_metrics']
    feature_matrix = data['feature_matrix']
    churn_predictions = data['churn']['predictions']
    segmentation_labels = data['segmentation']['labels']
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{kpi_metrics['total_customers']:,}", delta=None)
    
    with col2:
        st.metric("Total Revenue", f"${kpi_metrics['total_revenue']:,.0f}", delta=None)
    
    with col3:
        st.metric("Avg Transaction Value", f"${kpi_metrics['avg_transaction_value']:.2f}", delta=None)
    
    with col4:
        st.metric("Churn Rate", f"{kpi_metrics['churn_rate']:.1f}%", delta=None)
    
    st.markdown("---")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Customers", f"{kpi_metrics['active_customers']:,}", delta=None)
    
    with col2:
        st.metric("Avg Lifetime Value", f"${kpi_metrics['avg_lifetime_value']:,.0f}", delta=None)
    
    with col3:
        st.metric("Total Transactions", f"{kpi_metrics['total_transactions']:,}", delta=None)
    
    with col4:
        st.metric("Unique Products", f"{kpi_metrics['unique_products']:,}", delta=None)
    
    st.markdown("---")
    
    # Charts section
    st.markdown("### Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Segment Distribution")
        segment_dist = data['segment_distribution']
        
        fig = px.pie(
            values=segment_dist['Count'].values,
            names=segment_dist['Segment'].values,
            title="Customer Segments"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn Risk Distribution")
        risk_dist = data['churn_risk_distribution']
        risk_order = ['Low', 'Medium', 'High', 'Critical']
        risk_dist_ordered = risk_dist.set_index('RiskLevel').reindex([r for r in risk_order if r in risk_dist['RiskLevel'].values], fill_value=0).reset_index()
        
        fig = px.bar(
            risk_dist_ordered,
            x='RiskLevel',
            y='Count',
            title="Churn Risk Levels",
            labels={'Count': 'Number of Customers', 'RiskLevel': 'Risk Level'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue trend
    st.subheader("Revenue Trend Over Time")
    monthly_revenue = data['monthly_revenue']
    
    fig = px.line(
        monthly_revenue,
        x='YearMonth',
        y='LineTotal',
        title="Monthly Revenue Trend",
        labels={'LineTotal': 'Revenue ($)', 'YearMonth': 'Month'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("---")
    st.markdown("### Key Insights & Action Items")
    
    # Top segment
    top_segment_name = segment_dist.loc[segment_dist['Count'].idxmax(), 'Segment']
    top_segment_pct = segment_dist.loc[segment_dist['Count'].idxmax(), 'Percentage']
    
    # Critical churn risk
    critical_row = risk_dist[risk_dist['RiskLevel'] == 'Critical']
    critical_count = int(critical_row['Count'].iloc[0]) if len(critical_row) > 0 else 0
    critical_pct = float(critical_row['Percentage'].iloc[0]) if len(critical_row) > 0 else 0
    
    # High-value customers
    high_value_count = (feature_matrix['LifetimeValue'] > feature_matrix['LifetimeValue'].quantile(0.75)).sum() if 'LifetimeValue' in feature_matrix.columns else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Largest Customer Segment**  
        {top_segment_name}  
        {top_segment_pct:.1f}% of customers
        """)
    
    with col2:
        st.warning(f"""
        **Critical Churn Risk**  
        {critical_count} customers ({critical_pct:.1f}%)  
        Require immediate attention
        """)
    
    with col3:
        st.success(f"""
        **High-Value Customers**  
        {high_value_count} customers  
        Top 25% by lifetime value
        """)
    
    # Action recommendations
    st.markdown("#### Recommended Actions")
    action_items = []
    
    if critical_pct > 10:
        action_items.append(f"**Urgent**: {critical_count} customers at critical churn risk - implement retention campaigns immediately")
    
    if top_segment_pct > 40:
        action_items.append(f"**Focus**: {top_segment_name} represents {top_segment_pct:.1f}% of customers - develop targeted strategies for this segment")
    
    if 'LifetimeValue' in feature_matrix.columns:
        avg_ltv = feature_matrix['LifetimeValue'].mean()
        if avg_ltv < 1000:
            action_items.append("**Opportunity**: Average lifetime value is below target - focus on increasing customer value")
    
    if action_items:
        for item in action_items:
            st.markdown(f"- {item}")
    else:
        st.success("No critical action items at this time")


def show_customer_segments(data):
    """Display customer segments page."""
    st.header("Customer Segments")
    st.markdown("**Analyze customer segments and their characteristics**")
    st.markdown("---")
    
    segmentation_results = data['segmentation']
    feature_matrix = data['feature_matrix']
    labels = segmentation_results['labels']
    profiles = segmentation_results['profiles']
    segment_names = segmentation_results['segment_names']
    
    # Segment overview
    st.markdown("### Segment Overview")
    
    # Segment distribution (use pre-computed)
    segment_dist = data['segment_distribution']
    segment_counts = pd.Series(segment_dist['Count'].values, index=segment_dist['Segment'].values)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title="Customer Segment Distribution",
            labels={'x': 'Segment', 'y': 'Number of Customers'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Segment Sizes")
        for _, row in segment_dist.iterrows():
            st.metric(row['Segment'], f"{int(row['Count']):,}", f"{row['Percentage']:.1f}%")
    
    st.markdown("---")
    
    # Segment characteristics
    st.markdown("### Segment Characteristics")
    
    # Add segment names to feature matrix
    feature_with_segments = feature_matrix.copy()
    feature_with_segments['Segment'] = labels.values
    feature_with_segments['SegmentName'] = feature_with_segments['Segment'].map(segment_names)
    
    # Key metrics by segment
    key_metrics = ['Recency', 'Frequency', 'LifetimeValue', 'CurrentPointsBalance']
    available_metrics = [m for m in key_metrics if m in feature_with_segments.columns]
    
    if available_metrics:
        segment_metrics = feature_with_segments.groupby('SegmentName')[available_metrics].mean()
        
        st.subheader("Average Metrics by Segment")
        st.dataframe(segment_metrics.style.format("{:,.2f}"), use_container_width=True)
        
        # Visualization
        st.subheader("Segment Comparison")
        metric_to_plot = st.selectbox("Select metric to compare:", available_metrics)
        
        fig = px.bar(
            segment_metrics.reset_index(),
            x='SegmentName',
            y=metric_to_plot,
            title=f"{metric_to_plot} by Segment",
            labels={'SegmentName': 'Segment', metric_to_plot: metric_to_plot}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Segment details table
    st.markdown("### Detailed Segment Profiles")
    st.dataframe(profiles, use_container_width=True)


def show_churn_risk(data):
    """Display churn risk analysis page."""
    st.header("Churn Risk Analysis")
    st.markdown("**Identify at-risk customers and understand churn factors**")
    st.markdown("---")
    
    churn_results = data['churn']
    predictions = churn_results['predictions']
    metrics = churn_results['metrics']
    feature_importance = churn_results['feature_importance']
    
    # Model performance metrics
    st.markdown("### Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    with col2:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    with col3:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col4:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    
    st.markdown("---")
    
    # Churn risk distribution
    st.markdown("### Churn Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = predictions['RiskLevel'].value_counts()
        risk_order = ['Low', 'Medium', 'High', 'Critical']
        risk_counts = risk_counts.reindex([r for r in risk_order if r in risk_counts.index], fill_value=0)
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Churn Risk Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk level summary
        st.markdown("#### Risk Level Summary")
        for risk_level in risk_order:
            if risk_level in risk_counts.index:
                count = risk_counts[risk_level]
                pct = (count / len(predictions)) * 100
                st.metric(risk_level, f"{count:,}", f"{pct:.1f}%")
    
    st.markdown("---")
    
    # At-risk customers table
    st.markdown("### At-Risk Customers")
    st.markdown("**Review customers by churn risk level and take action**")
    
    predictions = data['churn']['predictions']
    risk_filter = st.selectbox("Filter by risk level:", ["All", "Critical", "High", "Medium", "Low"])
    
    if risk_filter == "All":
        at_risk = predictions.sort_values('ChurnProbability', ascending=False)
    else:
        at_risk = predictions[predictions['RiskLevel'] == risk_filter].sort_values('ChurnProbability', ascending=False)
    
    # Display summary
    st.info(f"Showing {len(at_risk)} customers with {risk_filter.lower()} risk level")
    
    # Format the dataframe for better display
    display_df = at_risk[['CustomerID', 'ChurnProbability', 'ChurnPrediction', 'RiskLevel']].head(100).copy()
    display_df['ChurnProbability'] = display_df['ChurnProbability'].apply(lambda x: f"{x:.2%}")
    display_df['ChurnPrediction'] = display_df['ChurnPrediction'].map({0: 'Active', 1: 'Churned'})
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download button
    csv = at_risk[['CustomerID', 'ChurnProbability', 'ChurnPrediction', 'RiskLevel']].to_csv(index=False)
    st.download_button(
        label="Download At-Risk Customers Data",
        data=csv,
        file_name=f"at_risk_customers_{risk_filter.lower()}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### Top Churn Risk Factors")
    
    feature_importance = data['churn']['feature_importance']
    top_features = feature_importance.head(15)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 15 Churn Risk Factors",
        labels={'Importance': 'Feature Importance', 'Feature': 'Feature Name'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def show_product_performance(data):
    """Display product performance page."""
    st.header("Product Performance")
    st.markdown("**Analyze product sales, revenue, and category performance**")
    st.markdown("---")
    
    # Top products by revenue (use pre-computed)
    st.markdown("### Top Products by Revenue")
    
    product_revenue = data['product_performance']
    
    # Display top N products
    num_products = st.slider("Number of top products to display:", 10, 50, 20)
    
    # Format for display
    display_df = product_revenue.head(num_products).copy()
    display_df['AvgPrice'] = (display_df['TotalRevenue'] / display_df['TotalQuantity']).round(2)
    
    # Format currency columns
    display_df_formatted = display_df.copy()
    display_df_formatted['TotalRevenue'] = display_df['TotalRevenue'].apply(lambda x: f"${x:,.2f}")
    display_df_formatted['AvgPrice'] = display_df['AvgPrice'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(display_df_formatted[['ProductName', 'Category', 'TotalRevenue', 'TotalQuantity', 'TransactionCount', 'AvgPrice']], 
                 use_container_width=True)
    
    # Visualization
    st.subheader("Top Products Visualization")
    fig = px.bar(
        product_revenue.head(num_products),
        x='TotalRevenue',
        y='ProductName',
        orientation='h',
        title=f"Top {num_products} Products by Revenue",
        labels={'TotalRevenue': 'Revenue ($)', 'ProductName': 'Product'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Category performance (use pre-computed)
    st.markdown("### Category Performance")
    
    category_perf = data['category_performance']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            category_perf,
            x='Category',
            y='TotalRevenue',
            title="Revenue by Category",
            labels={'TotalRevenue': 'Revenue ($)', 'Category': 'Category'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            category_perf,
            values='TotalRevenue',
            names='Category',
            title="Revenue Share by Category"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_loyalty_program(data):
    """Display loyalty program page."""
    st.header("Loyalty Program")
    st.markdown("**Monitor loyalty program engagement and tier distribution**")
    st.markdown("---")
    
    feature_matrix = data['feature_matrix']
    
    # Loyalty tier distribution
    st.markdown("### Loyalty Tier Distribution")
    
    if 'TierNumeric' in feature_matrix.columns:
        tier_dist = feature_matrix['TierNumeric'].value_counts().sort_index()
        tier_names = {1: 'Bronze', 2: 'Silver', 3: 'Gold', 4: 'Platinum'}
        tier_labels = [tier_names.get(i, f"Tier {i}") for i in tier_dist.index]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=tier_dist.values,
                names=tier_labels,
                title="Loyalty Tier Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Tier Summary")
            for tier_idx, tier_name in enumerate(tier_labels):
                count = tier_dist.iloc[tier_idx]
                pct = (count / len(feature_matrix)) * 100
                st.metric(tier_name, f"{count:,}", f"{pct:.1f}%")
    
    st.markdown("---")
    
    # Points statistics
    st.markdown("### Points Statistics")
    
    if 'CurrentPointsBalance' in feature_matrix.columns:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_points = feature_matrix['CurrentPointsBalance'].mean()
            st.metric("Avg Points Balance", f"{avg_points:,.0f}")
        
        with col2:
            total_points = feature_matrix['CurrentPointsBalance'].sum()
            st.metric("Total Points", f"{total_points:,.0f}")
        
        with col3:
            max_points = feature_matrix['CurrentPointsBalance'].max()
            st.metric("Max Points", f"{max_points:,.0f}")
        
        with col4:
            # Calculate total points earned from feature matrix if available
            if 'TotalPointsEarned' in feature_matrix.columns:
                total_points_earned = feature_matrix['TotalPointsEarned'].sum()
            else:
                total_points_earned = 0
            st.metric("Total Points Earned", f"{total_points_earned:,.0f}")


def show_findings_visualizations(data):
    """Display findings and visualizations page."""
    st.header("Findings & Visualizations")
    st.markdown("**Comprehensive visualizations and insights from the analysis**")
    st.markdown("---")
    
    visualizations_dir = Path("visualizations")
    
    if not visualizations_dir.exists():
        st.warning("""
        **Visualizations not found!**
        
        Please run the visualization generation script first:
        ```
        python generate_findings_visualizations.py
        ```
        """)
        return
    
    # Get all visualization files
    viz_files = {
        "Segment Distribution": "segment_distribution.png",
        "Segment RFM Comparison": "segment_rfm_comparison.png",
        "Segment Analysis": "segment_analysis.png",
        "Segment Lifetime Value": "segment_lifetime_value.png",
        "Churn Risk Distribution": "churn_risk_distribution.png",
        "Top Churn Risk Factors": "churn_risk_factors.png",
        "Model Performance": "model_performance.png",
        "Revenue Trend": "revenue_trend.png",
        "Category Performance": "category_performance.png",
        "KPI Summary": "kpi_summary.png"
    }
    
    # Section 1: Customer Segmentation
    st.markdown("## 1. Customer Segmentation Insights")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if (visualizations_dir / viz_files["Segment Distribution"]).exists():
            st.subheader("Segment Distribution")
            st.image(str(visualizations_dir / viz_files["Segment Distribution"]), 
                    use_container_width=True)
            st.caption("Distribution of customers across different segments identified through K-Means clustering.")
    
    with col2:
        if (visualizations_dir / viz_files["Segment RFM Comparison"]).exists():
            st.subheader("Segment RFM Characteristics")
            st.image(str(visualizations_dir / viz_files["Segment RFM Comparison"]), 
                    use_container_width=True)
            st.caption("Normalized comparison of Recency, Frequency, and Monetary values across segments.")
    
    if (visualizations_dir / viz_files["Segment Analysis"]).exists():
        st.subheader("Segment Analysis")
        st.image(str(visualizations_dir / viz_files["Segment Analysis"]), 
                use_container_width=True)
        st.caption("Left: Segment size vs average spend. Right: Frequency vs recency by segment.")
    
    if (visualizations_dir / viz_files["Segment Lifetime Value"]).exists():
        st.subheader("Average Lifetime Value by Segment")
        st.image(str(visualizations_dir / viz_files["Segment Lifetime Value"]), 
                use_container_width=True)
        st.caption("Comparison of average customer lifetime value across different segments.")
    
    st.markdown("---")
    
    # Section 2: Churn Prediction
    st.markdown("## 2. Churn Prediction Insights")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if (visualizations_dir / viz_files["Churn Risk Distribution"]).exists():
            st.subheader("Churn Risk Distribution")
            st.image(str(visualizations_dir / viz_files["Churn Risk Distribution"]), 
                    use_container_width=True)
            st.caption("Distribution of customers across different churn risk levels.")
    
    with col2:
        if (visualizations_dir / viz_files["Model Performance"]).exists():
            st.subheader("Model Performance Metrics")
            st.image(str(visualizations_dir / viz_files["Model Performance"]), 
                    use_container_width=True)
            st.caption("XGBoost churn prediction model performance metrics.")
    
    if (visualizations_dir / viz_files["Top Churn Risk Factors"]).exists():
        st.subheader("Top Churn Risk Factors")
        st.image(str(visualizations_dir / viz_files["Top Churn Risk Factors"]), 
                use_container_width=True)
        st.caption("Top 15 features that are most predictive of customer churn, ranked by importance.")
    
    st.markdown("---")
    
    # Section 3: Product Performance
    st.markdown("## 3. Product Performance Insights")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if (visualizations_dir / viz_files["Revenue Trend"]).exists():
            st.subheader("Monthly Revenue Trend")
            st.image(str(visualizations_dir / viz_files["Revenue Trend"]), 
                    use_container_width=True)
            st.caption("Revenue trends over time showing growth patterns and seasonality.")
    
    with col2:
        if (visualizations_dir / viz_files["Category Performance"]).exists():
            st.subheader("Category Performance")
            st.image(str(visualizations_dir / viz_files["Category Performance"]), 
                    use_container_width=True)
            st.caption("Revenue by product category with top 10 categories revenue share.")
    
    st.markdown("---")
    
    # Section 4: Key Performance Indicators
    st.markdown("## 4. Key Performance Indicators")
    st.markdown("---")
    
    if (visualizations_dir / viz_files["KPI Summary"]).exists():
        st.subheader("KPI Summary Dashboard")
        st.image(str(visualizations_dir / viz_files["KPI Summary"]), 
                use_container_width=True)
        st.caption("Summary of key performance indicators including revenue, customers, churn rate, and lifetime value.")
    
    st.markdown("---")
    
    # Additional Information
    st.markdown("## About These Visualizations")
    st.info("""
    These visualizations were generated using **matplotlib** and **seaborn** based on the comprehensive 
    analysis of customer engagement and loyalty data. They provide visual insights into:
    
    - **Customer Segmentation**: How customers are grouped based on behavior and value
    - **Churn Prediction**: Risk factors and model performance for identifying at-risk customers
    - **Product Performance**: Revenue trends and category analysis
    - **Key Metrics**: Overall business health indicators
    
    To regenerate these visualizations with updated data, run:
    ```bash
    python generate_findings_visualizations.py
    ```
    
    For detailed findings and recommendations, see the **FINDINGS.md** document.
    """)


if __name__ == "__main__":
    main()

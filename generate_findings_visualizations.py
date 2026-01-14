"""
Generate visualizations for the FINDINGS.md document.
This script creates graphs using seaborn and matplotlib and saves them as images.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

# Load data
cache_dir = Path("data/dashboard_cache")

print("Loading data...")
segment_profiles = pd.read_csv(cache_dir / "segment_profiles.csv")
segment_distribution = pd.read_csv(cache_dir / "segment_distribution.csv")
churn_risk_distribution = pd.read_csv(cache_dir / "churn_risk_distribution.csv")
churn_feature_importance = pd.read_csv(cache_dir / "churn_feature_importance.csv")
category_performance = pd.read_csv(cache_dir / "category_performance.csv")
monthly_revenue = pd.read_csv(cache_dir / "monthly_revenue.csv")
kpi_metrics = json.load(open(cache_dir / "kpi_metrics.json"))
churn_metrics = json.load(open(cache_dir / "churn_metrics.json"))

# Load segment names
with open(cache_dir / "segment_names.json", 'r') as f:
    segment_names = json.load(f)

print("Generating visualizations...")

# 1. Segment Distribution Pie Chart
print("1. Segment Distribution...")
fig, ax = plt.subplots(figsize=(10, 8))
colors = sns.color_palette("Set2", len(segment_distribution))
wedges, texts, autotexts = ax.pie(
    segment_distribution['Count'],
    labels=segment_distribution['Segment'],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90
)
ax.set_title("Customer Segment Distribution", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / "segment_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Segment Characteristics - RFM Comparison
print("2. Segment RFM Characteristics...")
# Prepare data for RFM comparison
rfm_data = []
for idx, row in segment_profiles.iterrows():
    segment = int(row['Segment'])
    segment_name = segment_names.get(str(segment), f"Segment {segment}")
    rfm_data.append({
        'Segment': segment_name,
        'Recency': row['AvgRecency'],
        'Frequency': row['AvgFrequency'],
        'Monetary': row['AvgTotalSpend'] / 1000  # Convert to thousands
    })
rfm_df = pd.DataFrame(rfm_data)

# Normalize for comparison (0-1 scale)
rfm_normalized = rfm_df.copy()
for col in ['Recency', 'Frequency', 'Monetary']:
    max_val = rfm_normalized[col].max()
    min_val = rfm_normalized[col].min()
    if max_val > min_val:
        rfm_normalized[col] = (rfm_normalized[col] - min_val) / (max_val - min_val)

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(rfm_normalized))
width = 0.25

ax.bar(x - width, rfm_normalized['Recency'], width, label='Recency (normalized)', color='#3498db')
ax.bar(x, rfm_normalized['Frequency'], width, label='Frequency (normalized)', color='#2ecc71')
ax.bar(x + width, rfm_normalized['Monetary'], width, label='Monetary (normalized)', color='#e74c3c')

ax.set_xlabel('Segment', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
ax.set_title('Segment RFM Characteristics Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(rfm_normalized['Segment'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "segment_rfm_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Churn Risk Distribution
print("3. Churn Risk Distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
risk_order = ['Low', 'Medium', 'High', 'Critical']
risk_dist_ordered = churn_risk_distribution.set_index('RiskLevel').reindex(
    [r for r in risk_order if r in churn_risk_distribution['RiskLevel'].values]
).reset_index()
risk_colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e67e22', 'Critical': '#e74c3c'}
colors_list = [risk_colors.get(risk, '#95a5a6') for risk in risk_dist_ordered['RiskLevel']]

wedges, texts, autotexts = ax1.pie(
    risk_dist_ordered['Count'],
    labels=risk_dist_ordered['RiskLevel'],
    autopct='%1.1f%%',
    colors=colors_list,
    startangle=90
)
ax1.set_title("Churn Risk Distribution", fontsize=14, fontweight='bold')

# Bar chart
ax2.bar(risk_dist_ordered['RiskLevel'], risk_dist_ordered['Count'], color=colors_list)
ax2.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
ax2.set_title("Churn Risk Counts", fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(risk_dist_ordered['Count']):
    ax2.text(i, v + 20, f'{int(v):,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "churn_risk_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Top Churn Risk Factors
print("4. Top Churn Risk Factors...")
top_factors = churn_feature_importance.head(15)
fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette("Reds_r", len(top_factors))
bars = ax.barh(top_factors['Feature'], top_factors['Importance'], color=colors)
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Churn Risk Factors', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, (idx, row) in enumerate(top_factors.iterrows()):
    ax.text(row['Importance'] + 0.005, i, f"{row['Importance']:.3f}", 
            va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "churn_risk_factors.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Revenue Trend Over Time
print("5. Revenue Trend...")
monthly_revenue['YearMonth'] = pd.to_datetime(monthly_revenue['YearMonth'], format='%Y-%m')
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(monthly_revenue['YearMonth'], monthly_revenue['LineTotal'] / 1e6, 
        marker='o', linewidth=2, markersize=6, color='#3498db')
ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Revenue (Millions $)', fontsize=12, fontweight='bold')
ax.set_title('Monthly Revenue Trend', fontsize=16, fontweight='bold', pad=20)
ax.grid(alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / "revenue_trend.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Category Performance
print("6. Category Performance...")
category_perf_sorted = category_performance.sort_values('TotalRevenue', ascending=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Horizontal bar chart
colors = sns.color_palette("viridis", len(category_perf_sorted))
bars = ax1.barh(category_perf_sorted['Category'], category_perf_sorted['TotalRevenue'] / 1e6, color=colors)
ax1.set_xlabel('Revenue (Millions $)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Category', fontsize=12, fontweight='bold')
ax1.set_title('Revenue by Category', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
# Add value labels
for i, (idx, row) in enumerate(category_perf_sorted.iterrows()):
    ax1.text(row['TotalRevenue'] / 1e6 + 0.5, i, f"${row['TotalRevenue']/1e6:.1f}M", 
            va='center', fontweight='bold')

# Pie chart (top 10 categories)
top_categories = category_perf_sorted.tail(10)
wedges, texts, autotexts = ax2.pie(
    top_categories['TotalRevenue'],
    labels=top_categories['Category'],
    autopct='%1.1f%%',
    startangle=90
)
ax2.set_title('Top 10 Categories Revenue Share', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "category_performance.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Segment Size vs Average Spend
print("7. Segment Size vs Spend...")
segment_metrics = []
for idx, row in segment_profiles.iterrows():
    segment = int(row['Segment'])
    segment_name = segment_names.get(str(segment), f"Segment {segment}")
    segment_metrics.append({
        'Segment': segment_name,
        'Size': row['SegmentSize'],
        'AvgSpend': row['AvgTotalSpend'],
        'AvgFrequency': row['AvgFrequency'],
        'AvgRecency': row['AvgRecency']
    })
seg_metrics_df = pd.DataFrame(segment_metrics)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Size vs Spend scatter
scatter = ax1.scatter(seg_metrics_df['Size'], seg_metrics_df['AvgSpend'] / 1000, 
                     s=200, alpha=0.6, c=range(len(seg_metrics_df)), cmap='viridis')
for i, row in seg_metrics_df.iterrows():
    ax1.annotate(row['Segment'], (row['Size'], row['AvgSpend'] / 1000),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax1.set_xlabel('Segment Size (Number of Customers)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Spend (Thousands $)', fontsize=12, fontweight='bold')
ax1.set_title('Segment Size vs Average Spend', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# Frequency vs Recency
scatter = ax2.scatter(seg_metrics_df['AvgFrequency'], seg_metrics_df['AvgRecency'],
                     s=200, alpha=0.6, c=range(len(seg_metrics_df)), cmap='plasma')
for i, row in seg_metrics_df.iterrows():
    ax2.annotate(row['Segment'], (row['AvgFrequency'], row['AvgRecency']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax2.set_xlabel('Average Frequency', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Recency (Days)', fontsize=12, fontweight='bold')
ax2.set_title('Frequency vs Recency by Segment', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "segment_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. Model Performance Metrics
print("8. Model Performance...")
metrics_data = {
    'Metric': ['ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Accuracy'],
    'Value': [
        churn_metrics['roc_auc'],
        churn_metrics['f1_score'],
        churn_metrics['precision'],
        churn_metrics['recall'],
        churn_metrics['accuracy']
    ]
}
metrics_df = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#9b59b6']
bars = ax.bar(metrics_df['Metric'], metrics_df['Value'], color=colors)
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Churn Prediction Model Performance', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
# Add value labels
for i, (idx, row) in enumerate(metrics_df.iterrows()):
    ax.text(i, row['Value'] + 0.02, f"{row['Value']:.3f}", 
            ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / "model_performance.png", dpi=300, bbox_inches='tight')
plt.close()

# 9. KPI Summary Dashboard
print("9. KPI Summary...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Key Performance Indicators Summary', fontsize=18, fontweight='bold', y=0.98)

# Total Revenue
axes[0, 0].text(0.5, 0.5, f"${kpi_metrics['total_revenue']/1e6:.1f}M", 
                ha='center', va='center', fontsize=24, fontweight='bold')
axes[0, 0].text(0.5, 0.3, 'Total Revenue', ha='center', va='center', fontsize=14)
axes[0, 0].axis('off')

# Total Customers
axes[0, 1].text(0.5, 0.5, f"{kpi_metrics['total_customers']:,}", 
                ha='center', va='center', fontsize=24, fontweight='bold')
axes[0, 1].text(0.5, 0.3, 'Total Customers', ha='center', va='center', fontsize=14)
axes[0, 1].axis('off')

# Active Customers
axes[0, 2].text(0.5, 0.5, f"{kpi_metrics['active_customers']:,}", 
                ha='center', va='center', fontsize=24, fontweight='bold', color='#2ecc71')
axes[0, 2].text(0.5, 0.3, f"Active Customers ({100*kpi_metrics['active_customers']/kpi_metrics['total_customers']:.1f}%)", 
                ha='center', va='center', fontsize=14)
axes[0, 2].axis('off')

# Churn Rate
axes[1, 0].text(0.5, 0.5, f"{kpi_metrics['churn_rate']:.2f}%", 
                ha='center', va='center', fontsize=24, fontweight='bold', color='#e74c3c')
axes[1, 0].text(0.5, 0.3, 'Churn Rate', ha='center', va='center', fontsize=14)
axes[1, 0].axis('off')

# Avg Transaction Value
axes[1, 1].text(0.5, 0.5, f"${kpi_metrics['avg_transaction_value']:.2f}", 
                ha='center', va='center', fontsize=24, fontweight='bold')
axes[1, 1].text(0.5, 0.3, 'Avg Transaction Value', ha='center', va='center', fontsize=14)
axes[1, 1].axis('off')

# Avg Lifetime Value
axes[1, 2].text(0.5, 0.5, f"${kpi_metrics['avg_lifetime_value']/1000:.1f}K", 
                ha='center', va='center', fontsize=24, fontweight='bold')
axes[1, 2].text(0.5, 0.3, 'Avg Lifetime Value', ha='center', va='center', fontsize=14)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "kpi_summary.png", dpi=300, bbox_inches='tight')
plt.close()

# 10. Segment Lifetime Value Comparison
print("10. Segment Lifetime Value...")
fig, ax = plt.subplots(figsize=(12, 6))
segments_sorted = segment_profiles.sort_values('AvgLifetimeValue', ascending=True)
segment_labels = [segment_names.get(str(int(s)), f"Segment {int(s)}") for s in segments_sorted['Segment']]
colors = sns.color_palette("coolwarm", len(segments_sorted))
bars = ax.barh(segment_labels, segments_sorted['AvgLifetimeValue'] / 1000, color=colors)
ax.set_xlabel('Average Lifetime Value (Thousands $)', fontsize=12, fontweight='bold')
ax.set_ylabel('Segment', fontsize=12, fontweight='bold')
ax.set_title('Average Lifetime Value by Segment', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
# Add value labels
for i, (idx, row) in enumerate(segments_sorted.iterrows()):
    ax.text(row['AvgLifetimeValue'] / 1000 + 5, i, f"${row['AvgLifetimeValue']/1000:.1f}K", 
            va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "segment_lifetime_value.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to {output_dir}/")
print("Generated files:")
for file in sorted(output_dir.glob("*.png")):
    print(f"  - {file.name}")

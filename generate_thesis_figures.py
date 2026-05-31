"""
Thesis Figure Generator — Heart Disease Risk Prediction Project
Generates high-quality PNG figures for the thesis document.
Output directory: thesis_figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart_disease_uci.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUT_DIR = os.path.join(BASE_DIR, "thesis_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#CCCCCC',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FF9800',
    'purple': '#9C27B0',
    'teal': '#009688',
    'healthy': '#4CAF50',
    'at_risk': '#F44336',
}
MODEL_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']


def load_data():
    """Load raw dataset"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    return df


def load_results():
    """Load model training results"""
    print("Loading model artifacts...")
    try:
        all_results = joblib.load(os.path.join(ARTIFACTS_DIR, "all_results.pkl"))
        feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))
        model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
        return all_results, feature_names, model
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_model.py first!")
        return None, None, None


# FIGURE 1: Target Distribution
def fig1_target_distribution(df):
    print("Generating: Target Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    counts = df['target'].value_counts()
    labels = ['Healthy (0)', 'At Risk (1+)']
    colors = [COLORS['healthy'], COLORS['at_risk']]
    explode = (0, 0.05)
    
    wedges, texts, autotexts = axes[0].pie(
        counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
        explode=explode, startangle=90, textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontweight('bold')
    axes[0].set_title('Target Variable Distribution')

    # Bar chart
    bars = axes[1].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=2, width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha='center', va='bottom', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Class Counts')
    axes[1].set_ylim(0, max(counts.values) * 1.15)

    plt.suptitle('Figure 1: Heart Disease Target Variable Distribution', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_target_distribution.png"))
    plt.close()


# FIGURE 2: Correlation Heatmap
def fig2_correlation_heatmap(df):
    print("Generating: Correlation Heatmap...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    im = ax.imshow(corr.where(~mask).values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)
    
    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            if not mask[i, j]:
                val = corr.iloc[i, j]
                color = 'white' if abs(val) > 0.4 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=11)
    
    ax.set_title('Figure 2: Feature Correlation Matrix', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_correlation_heatmap.png"))
    plt.close()


# FIGURE 3: Feature Distributions by Target
def fig3_feature_distributions(df):
    print("Generating: Feature Distributions...")
    features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    titles = ['Age', 'Resting Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Depression']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (feat, title) in enumerate(zip(features, titles)):
        ax = axes[i]
        healthy = df[df['target'] == 0][feat].dropna()
        at_risk = df[df['target'] == 1][feat].dropna()

        bp1 = ax.boxplot([healthy, at_risk], labels=['Healthy', 'At Risk'],
                         patch_artist=True, widths=0.5,
                         medianprops=dict(color='black', linewidth=2))
        
        bp1['boxes'][0].set_facecolor(COLORS['healthy'])
        bp1['boxes'][0].set_alpha(0.7)
        bp1['boxes'][1].set_facecolor(COLORS['at_risk'])
        bp1['boxes'][1].set_alpha(0.7)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Value')

    # Remove empty subplot
    axes[5].axis('off')

    plt.suptitle('Figure 3: Feature Distributions by Heart Disease Status', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_feature_distributions.png"))
    plt.close()


# FIGURE 4: Missing Values
def fig4_missing_values(df):
    print("Generating: Missing Values...")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if len(missing) == 0:
        print("  No missing values found, skipping...")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(missing.index, missing.values, color=COLORS['warning'], edgecolor='white', height=0.6)
    
    for bar, val in zip(bars, missing.values):
        pct = val / len(df) * 100
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{val} ({pct:.1f}%)', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Missing Values')
    ax.set_title('Figure 4: Missing Values by Feature', fontsize=15, fontweight='bold')
    ax.set_xlim(0, max(missing.values) * 1.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_missing_values.png"))
    plt.close()


# FIGURE 5: Model Comparison (Bar Chart)
def fig5_model_comparison(all_results):
    print("Generating: Model Comparison...")
    metrics_list = all_results['metrics']
    df_metrics = pd.DataFrame(metrics_list)

    models = df_metrics['Model'].tolist()
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(models))
    width = 0.15
    bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    
    for i, (metric, color) in enumerate(zip(metric_cols, bar_colors)):
        values = df_metrics[metric].values
        bars = ax.bar(x + i * width, values, width, label=metric, color=color, edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{val*100:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Figure 5: Model Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', fontsize=9, ncol=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_model_comparison.png"))
    plt.close()


# FIGURE 6: ROC Curves
def fig6_roc_curves(all_results):
    print("Generating: ROC Curves...")
    roc_data = all_results['roc_data']

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (name, data) in enumerate(roc_data.items()):
        ax.plot(data['fpr'], data['tpr'],
                label=f"{name} (AUC = {data['auc']:.3f})",
                color=MODEL_COLORS[i % len(MODEL_COLORS)],
                linewidth=2.5)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Figure 6: ROC Curves — All Models', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_roc_curves.png"))
    plt.close()


# FIGURE 7: Confusion Matrices
def fig7_confusion_matrices(all_results):
    print("Generating: Confusion Matrices...")
    cm_data = all_results['confusion_matrices']
    n_models = len(cm_data)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cm_data.items()):
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Healthy', 'At Risk'], fontsize=10)
        ax.set_yticklabels(['Healthy', 'At Risk'], fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(name, fontweight='bold', fontsize=12)
        
        # Add text values
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=18, fontweight='bold', color=color)

    plt.suptitle('Figure 7: Confusion Matrices — All Models', fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig7_confusion_matrices.png"))
    plt.close()


# FIGURE 8: Feature Importance (from model)
def fig8_feature_importance(model, feature_names):
    print("Generating: Feature Importance...")
    
    # Try to get feature importance from the model
    base_model = model
    if hasattr(model, 'estimators_'):
        # VotingClassifier - use first estimator (Random Forest)
        base_model = model.estimators_[0]
    
    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
    else:
        print("  Model does not have feature_importances_, skipping...")
        return

    # Sort by importance
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_values = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_values)))
    
    bars = ax.barh(range(len(sorted_names)), sorted_values, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Figure 8: Feature Importance — Random Forest', fontsize=15, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, sorted_values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_feature_importance.png"))
    plt.close()


# FIGURE 9: Age Distribution by Target
def fig9_age_distribution(df):
    print("Generating: Age Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))

    healthy = df[df['target'] == 0]['age'].dropna()
    at_risk = df[df['target'] == 1]['age'].dropna()

    ax.hist(healthy, bins=20, alpha=0.7, color=COLORS['healthy'], label='Healthy', edgecolor='white')
    ax.hist(at_risk, bins=20, alpha=0.7, color=COLORS['at_risk'], label='At Risk', edgecolor='white')
    
    ax.axvline(healthy.mean(), color=COLORS['healthy'], linestyle='--', linewidth=2, label=f'Healthy Mean: {healthy.mean():.1f}')
    ax.axvline(at_risk.mean(), color=COLORS['at_risk'], linestyle='--', linewidth=2, label=f'At Risk Mean: {at_risk.mean():.1f}')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.set_title('Figure 9: Age Distribution by Heart Disease Status', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig9_age_distribution.png"))
    plt.close()


# FIGURE 10: Metrics Summary Table (as image)
def fig10_metrics_table(all_results):
    print("Generating: Metrics Table...")
    metrics_list = all_results['metrics']
    df_m = pd.DataFrame(metrics_list)
    
    # Format for display
    display_df = df_m.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        display_df[col] = display_df[col].apply(lambda x: f'{x*100:.2f}%')
    
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for j, col in enumerate(display_df.columns):
        cell = table[0, j]
        cell.set_facecolor('#2196F3')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Style best model row (first row since sorted by ROC-AUC)
    for j in range(len(display_df.columns)):
        cell = table[1, j]
        cell.set_facecolor('#E8F5E9')
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            cell = table[i, j]
            if i == 1:
                cell.set_facecolor('#E8F5E9')
            elif i % 2 == 0:
                cell.set_facecolor('#F5F5F5')
    
    ax.set_title('Figure 10: Model Performance Metrics Summary', fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig10_metrics_table.png"))
    plt.close()


# MAIN
def main():
    print("=" * 50)
    print("  Thesis Figure Generator")
    print("=" * 50)
    print()
    
    # Load data
    df = load_data()
    all_results, feature_names, model = load_results()
    
    if all_results is None:
        print("\nCannot generate model figures without training results.")
        print("Generating data-only figures...\n")
    
    # Generate all figures
    fig1_target_distribution(df)
    fig2_correlation_heatmap(df)
    fig3_feature_distributions(df)
    fig4_missing_values(df)
    fig9_age_distribution(df)
    
    if all_results is not None:
        fig5_model_comparison(all_results)
        fig6_roc_curves(all_results)
        fig7_confusion_matrices(all_results)
        fig10_metrics_table(all_results)
    
    if model is not None and feature_names is not None:
        fig8_feature_importance(model, feature_names)
    
    print()
    print("=" * 50)
    print(f"  All figures saved to: {OUTPUT_DIR}")
    print("=" * 50)
    print()
    print("Generated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  📊 {f}")


if __name__ == "__main__":
    main()

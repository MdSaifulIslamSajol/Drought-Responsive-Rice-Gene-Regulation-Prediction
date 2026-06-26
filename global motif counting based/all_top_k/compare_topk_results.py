# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Top-K Varieties Performance Comparison and Visualization
=========================================================
This script aggregates results from all top-k experiments and creates
comprehensive comparison plots to determine optimal number of varieties.

Usage:
    python compare_topk_results.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = "/home/saiful/DeepRice_Task1/motif_based_classification_ALL_DATA/MSc_thesis2/global motif counting based/all_top_k"
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_results")
DPI = 400

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


def load_all_results():
    """Load results from all top-k experiments."""
    all_results = {}
    
    for k in range(1, 14):  # top1 to top13
        results_path = os.path.join(BASE_DIR, f"top{k}", "results", "results_summary.json")
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                all_results[k] = json.load(f)
            print(f"  Loaded top{k} results")
        else:
            print(f"  top{k} results not found")
    
    return all_results


def create_comparison_dataframe(all_results):
    """Create a DataFrame with key metrics for all top-k configurations."""
    rows = []
    
    for k, data in all_results.items():
        best_model = data['best_model']
        best_results = data['results'][best_model]['test']
        
        row = {
            'top_k': k,
            'n_varieties': data['data_stats']['num_varieties'],
            'variety_names': ', '.join(data['data_stats']['variety_names']),
            'train_samples': data['data_stats']['train_samples'],
            'dev_samples': data['data_stats']['dev_samples'],
            'test_samples': data['data_stats']['test_samples'],
            'total_samples': data['data_stats']['train_samples'] + data['data_stats']['dev_samples'] + data['data_stats']['test_samples'],
            'best_model': best_model,
            'test_auc': best_results['auc'],
            'test_f1': best_results['f1'],
            'test_mcc': best_results['mcc'],
            'test_balanced_accuracy': best_results['balanced_accuracy'],
            'test_precision': best_results['precision'],
            'test_recall': best_results['recall'],
            'test_specificity': best_results['specificity'],
            'test_pr_auc': best_results['pr_auc'],
        }
        
        # Add all models' results
        for model_name, model_res in data['results'].items():
            row[f'{model_name}_auc'] = model_res['test']['auc']
            row[f'{model_name}_f1'] = model_res['test']['f1']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('top_k')
    return df


def plot_performance_vs_varieties(df, output_dir):
    """Plot performance metrics vs number of varieties."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Main performance metrics vs top-k
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics = [
        ('test_auc', 'ROC-AUC', 'steelblue'),
        ('test_f1', 'F1 Score', 'darkorange'),
        ('test_mcc', 'MCC', 'forestgreen'),
        ('test_balanced_accuracy', 'Balanced Accuracy', 'purple')
    ]
    
    for idx, (metric, label, color) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        ax.plot(df['top_k'], df[metric], 'o-', color=color, linewidth=2.5, markersize=10)
        ax.fill_between(df['top_k'], df[metric], alpha=0.2, color=color)
        
        # Find best k
        best_idx = df[metric].idxmax()
        best_k = df.loc[best_idx, 'top_k']
        best_val = df.loc[best_idx, metric]
        ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best: k={best_k}')
        ax.scatter([best_k], [best_val], s=200, c='red', zorder=5, marker='*')
        
        ax.set_xlabel('Number of Top Varieties (k)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Number of Varieties', fontweight='bold')
        ax.set_xticks(df['top_k'])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Annotate values
        for i, row in df.iterrows():
            ax.annotate(f'{row[metric]:.3f}', (row["top_k"], row[metric]),
                       textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    
    plt.suptitle('Classification Performance vs Number of Rice Varieties', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig1_performance_vs_varieties.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig1_performance_vs_varieties.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Sample size vs performance
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    ax2 = ax1.twinx()
    
    bars = ax1.bar(df['top_k'], df['total_samples'], color='lightblue', alpha=0.7, label='Total Samples')
    ax1.set_xlabel('Number of Top Varieties (k)')
    ax1.set_ylabel('Total Samples', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(df['top_k'])
    
    line = ax2.plot(df['top_k'], df['test_auc'], 'ro-', linewidth=2.5, markersize=10, label='Test AUC')
    ax2.set_ylabel('Test AUC', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.title('Sample Size and Performance vs Number of Varieties', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_samples_vs_performance.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig2_samples_vs_performance.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Heatmap of all models across all k values
    model_names = ['HybridMLP', 'HybridLSTM', 'HybridTransformer', 'HybridCNNAttention', 
                   'HybridGatedFusion', 'CrossAttentionFusion', 'HybridMultiScaleCNN']
    
    auc_matrix = np.zeros((len(df), len(model_names)))
    for i, row in df.iterrows():
        for j, model in enumerate(model_names):
            col_name = f'{model}_auc'
            if col_name in row:
                auc_matrix[df.index.get_loc(i), j] = row[col_name]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(auc_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                xticklabels=[m.replace('Hybrid', '') for m in model_names],
                yticklabels=[f'Top {k}' for k in df['top_k']],
                cbar_kws={'label': 'Test AUC'},
                vmin=0.5, vmax=0.85)
    
    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Number of Top Varieties')
    ax.set_title('Test AUC Heatmap: All Models × All Top-K Configurations', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_auc_heatmap.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_auc_heatmap.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Best model per k
    fig, ax = plt.subplots(figsize=(12, 6))
    
    best_models = df['best_model'].values
    unique_models = list(set(best_models))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_models)))
    model_color_map = {m: colors[i] for i, m in enumerate(unique_models)}
    
    bar_colors = [model_color_map[m] for m in best_models]
    bars = ax.bar(df['top_k'], df['test_auc'], color=bar_colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Number of Top Varieties (k)')
    ax.set_ylabel('Test AUC')
    ax.set_title('Best Model Performance per Top-K Configuration', fontweight='bold')
    ax.set_xticks(df['top_k'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add model name annotations
    for i, (bar, model) in enumerate(zip(bars, best_models)):
        height = bar.get_height()
        ax.annotate(model.replace('Hybrid', '').replace('Fusion', ''),
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=model_color_map[m]) for m in unique_models]
    ax.legend(handles, [m.replace('Hybrid', '') for m in unique_models], 
              loc='lower right', title='Best Model')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_best_model_per_k.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig4_best_model_per_k.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Multi-metric comparison (radar chart for selected k values)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    
    selected_k = [df['top_k'].min(), df.loc[df['test_auc'].idxmax(), 'top_k'], df['top_k'].max()]
    categories = ['AUC', 'F1', 'MCC', 'Bal.Acc', 'Precision', 'Recall']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for idx, k in enumerate(selected_k):
        ax = axes[idx]
        row = df[df['top_k'] == k].iloc[0]
        
        values = [row['test_auc'], row['test_f1'], row['test_mcc'], 
                  row['test_balanced_accuracy'], row['test_precision'], row['test_recall']]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        
        title = f'Top {k} Varieties'
        if k == selected_k[1]:
            title += ' (BEST)'
        ax.set_title(title, fontweight='bold', size=12, pad=20)
    
    plt.suptitle('Performance Profiles Across Top-K Configurations', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig5_radar_comparison.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig5_radar_comparison.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  All figures saved to {output_dir}")


def generate_report(df, output_dir):
    """Generate comprehensive text report."""
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("TOP-K RICE VARIETIES CLASSIFICATION COMPARISON REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Best overall configuration
        best_idx = df['test_auc'].idxmax()
        best_row = df.loc[best_idx]
        
        f.write("-" * 100 + "\n")
        f.write("BEST OVERALL CONFIGURATION\n")
        f.write("-" * 100 + "\n")
        f.write(f"  Optimal Number of Varieties: Top {int(best_row['top_k'])}\n")
        f.write(f"  Varieties: {best_row['variety_names']}\n")
        f.write(f"  Best Model: {best_row['best_model']}\n")
        f.write(f"  Total Samples: {int(best_row['total_samples']):,}\n")
        f.write(f"\n  Performance Metrics:\n")
        f.write(f"    - ROC-AUC:           {best_row['test_auc']:.4f}\n")
        f.write(f"    - F1 Score:          {best_row['test_f1']:.4f}\n")
        f.write(f"    - MCC:               {best_row['test_mcc']:.4f}\n")
        f.write(f"    - Balanced Accuracy: {best_row['test_balanced_accuracy']:.4f}\n")
        f.write(f"    - Precision:         {best_row['test_precision']:.4f}\n")
        f.write(f"    - Recall:            {best_row['test_recall']:.4f}\n")
        f.write(f"    - Specificity:       {best_row['test_specificity']:.4f}\n")
        f.write(f"    - PR-AUC:            {best_row['test_pr_auc']:.4f}\n")
        
        # Performance ranking
        f.write("\n" + "-" * 100 + "\n")
        f.write("PERFORMANCE RANKING (By Test AUC)\n")
        f.write("-" * 100 + "\n")
        
        df_sorted = df.sort_values('test_auc', ascending=False)
        f.write(f"\n{'Rank':<6} {'Top-K':<8} {'Model':<25} {'Samples':<12} {'AUC':<10} {'F1':<10} {'MCC':<10}\n")
        f.write("-" * 91 + "\n")
        
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            f.write(f"{rank:<6} Top {int(row['top_k']):<4} {row['best_model']:<25} "
                   f"{int(row['total_samples']):<12,} {row['test_auc']:<10.4f} "
                   f"{row['test_f1']:<10.4f} {row['test_mcc']:<10.4f}\n")
        
        # Worst performing configuration
        worst_idx = df['test_auc'].idxmin()
        worst_row = df.loc[worst_idx]
        
        f.write("\n" + "-" * 100 + "\n")
        f.write("WORST PERFORMING CONFIGURATION\n")
        f.write("-" * 100 + "\n")
        f.write(f"  Configuration: Top {int(worst_row['top_k'])}\n")
        f.write(f"  Best Model: {worst_row['best_model']}\n")
        f.write(f"  Test AUC: {worst_row['test_auc']:.4f}\n")
        f.write(f"  Test F1: {worst_row['test_f1']:.4f}\n")
        
        # Summary statistics
        f.write("\n" + "-" * 100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 100 + "\n")
        f.write(f"  Total configurations tested: {len(df)}\n")
        f.write(f"  AUC range: {df['test_auc'].min():.4f} - {df['test_auc'].max():.4f}\n")
        f.write(f"  AUC mean ± std: {df['test_auc'].mean():.4f} ± {df['test_auc'].std():.4f}\n")
        f.write(f"  F1 range: {df['test_f1'].min():.4f} - {df['test_f1'].max():.4f}\n")
        f.write(f"  MCC range: {df['test_mcc'].min():.4f} - {df['test_mcc'].max():.4f}\n")
        
        # Variety details table
        f.write("\n" + "-" * 100 + "\n")
        f.write("VARIETY DETAILS BY TOP-K\n")
        f.write("-" * 100 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"\nTop {int(row['top_k'])}: {row['variety_names']}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"  Report saved to {report_path}")
    return report_path


def main():
    print("=" * 80)
    print("TOP-K VARIETIES COMPARISON")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all results
    print("\nLoading results...")
    all_results = load_all_results()
    
    if len(all_results) == 0:
        print("\nERROR: No results found. Please run the pipeline first.")
        return
    
    print(f"\nFound {len(all_results)} top-k configurations")
    
    # Create comparison DataFrame
    print("\nCreating comparison DataFrame...")
    df = create_comparison_dataframe(all_results)
    
    # Save DataFrame
    csv_path = os.path.join(OUTPUT_DIR, 'comparison_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Data saved to {csv_path}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_performance_vs_varieties(df, OUTPUT_DIR)
    
    # Generate report
    print("\nGenerating report...")
    report_path = generate_report(df, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    
    best_idx = df['test_auc'].idxmax()
    best_row = df.loc[best_idx]
    worst_idx = df['test_auc'].idxmin()
    worst_row = df.loc[worst_idx]
    
    print(f"\n  BEST Configuration:  Top {int(best_row['top_k'])} varieties (AUC={best_row['test_auc']:.4f})")
    print(f"    Model: {best_row['best_model']}")
    print(f"    Varieties: {best_row['variety_names']}")
    
    print(f"\n  WORST Configuration: Top {int(worst_row['top_k'])} varieties (AUC={worst_row['test_auc']:.4f})")
    print(f"    Model: {worst_row['best_model']}")
    
    print(f"\n  AUC Improvement: {(best_row['test_auc'] - worst_row['test_auc']):.4f} "
          f"({(best_row['test_auc'] - worst_row['test_auc']) / worst_row['test_auc'] * 100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

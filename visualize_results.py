import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load all the analysis results."""
    results_df = pd.read_csv("spearman_vs_gold_summary.csv")
    global_corr_df = pd.read_csv("global_model_correlations.csv", index_col=0)
    rankings_df = pd.read_csv("labse_proxy_rankings.csv")
    relative_perf_df = pd.read_csv("relative_performance_vs_labse.csv", index_col=0)
    
    return results_df, global_corr_df, rankings_df, relative_perf_df

def create_labse_correlation_chart(global_corr_df):
    """Create a bar chart showing correlation with LaBSE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get LaBSE correlations (excluding LaBSE itself)
    labse_corr = global_corr_df['labse'].drop('labse').sort_values(ascending=True)
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(labse_corr)), labse_corr.values, 
                   color=['#2E8B57', '#4169E1', '#FF6347'])
    
    # Customize the chart
    ax.set_yticks(range(len(labse_corr)))
    ax.set_yticklabels([name.replace('sbert_', '').replace('_', '-') for name in labse_corr.index])
    ax.set_xlabel('Correlation with LaBSE', fontsize=12, fontweight='bold')
    ax.set_title('🎯 Best LaBSE Proxy Models\n(Global Correlation Across 3,500 Text Pairs)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, labse_corr.values)):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontweight='bold')
        
        # Add ranking badges
        rank_colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
        rank_text = ['🥇 PRIMARY', '🥈 SPEED-OPT', '🥉 SPECIALIZED']
        ax.text(0.02, bar.get_y() + bar.get_height()/2, 
                rank_text[len(labse_corr)-1-i], va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=rank_colors[len(labse_corr)-1-i], alpha=0.8),
                fontsize=9, fontweight='bold')
    
    ax.set_xlim(0, 0.85)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('labse_correlation_ranking.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_heatmap(results_df):
    """Create a heatmap showing performance across datasets."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Pivot the data
    pivot_data = results_df.pivot(index='dataset', columns='model', values='spearman_vs_gold')
    
    # Reorder columns for better visualization
    column_order = ['labse', 'sbert_mpnet', 'sbert_minilm', 'gte_base']
    pivot_data = pivot_data[column_order]
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                cbar_kws={'label': 'Spearman Correlation'}, ax=ax)
    
    ax.set_title('📊 Model Performance Across Datasets\n(Spearman Correlation vs Gold Standard)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Datasets', fontsize=12, fontweight='bold')
    
    # Rename columns for display
    new_labels = ['LaBSE\n(Reference)', 'all-mpnet-base-v2\n(Best Proxy)', 
                  'all-MiniLM-L6-v2\n(Speed Optimized)', 'gte-base\n(Specialized)']
    ax.set_xticklabels(new_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_matrix(global_corr_df):
    """Create a correlation matrix between all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(global_corr_df, dtype=bool))
    
    # Create heatmap
    sns.heatmap(global_corr_df, mask=mask, annot=True, fmt='.3f', 
                cmap='coolwarm', center=0.7, square=True,
                cbar_kws={'label': 'Spearman Correlation'}, ax=ax)
    
    ax.set_title('🔗 Inter-Model Correlation Matrix\n(How Similar Are the Models?)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Rename labels
    labels = ['LaBSE', 'all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'gte-base']
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_dataset_type_comparison(results_df):
    """Compare performance by dataset type (sentence vs word)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sentence-level tasks
    sentence_datasets = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    sentence_data = results_df[results_df['dataset'].isin(sentence_datasets)]
    sentence_avg = sentence_data.groupby('model')['spearman_vs_gold'].mean().sort_values(ascending=False)
    
    bars1 = ax1.bar(range(len(sentence_avg)), sentence_avg.values, 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('📝 Sentence-Level Tasks\n(Average Performance)', fontweight='bold')
    ax1.set_ylabel('Average Spearman Correlation', fontweight='bold')
    ax1.set_xticks(range(len(sentence_avg)))
    ax1.set_xticklabels([name.replace('sbert_', '').replace('_', '-') for name in sentence_avg.index], 
                        rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars1, sentence_avg.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Word-level tasks
    word_data = results_df[results_df['dataset'] == 'SimLex-999']
    word_perf = word_data.set_index('model')['spearman_vs_gold'].sort_values(ascending=False)
    
    bars2 = ax2.bar(range(len(word_perf)), word_perf.values,
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('🔤 Word-Level Tasks\n(SimLex-999 Performance)', fontweight='bold')
    ax2.set_ylabel('Spearman Correlation', fontweight='bold')
    ax2.set_xticks(range(len(word_perf)))
    ax2.set_xticklabels([name.replace('sbert_', '').replace('_', '-') for name in word_perf.index], 
                        rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars2, word_perf.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_recommendation_dashboard(rankings_df, global_corr_df):
    """Create a comprehensive recommendation dashboard."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top left: LaBSE correlation ranking
    ax1 = fig.add_subplot(gs[0, 0])
    labse_corr = global_corr_df['labse'].drop('labse').sort_values(ascending=True)
    bars1 = ax1.barh(range(len(labse_corr)), labse_corr.values, 
                     color=['#2E8B57', '#4169E1', '#FF6347'])
    ax1.set_yticks(range(len(labse_corr)))
    ax1.set_yticklabels([name.replace('sbert_', '').replace('_', '-') for name in labse_corr.index])
    ax1.set_title('🎯 LaBSE Correlation', fontweight='bold')
    ax1.set_xlabel('Correlation')
    
    for bar, value in zip(bars1, labse_corr.values):
        ax1.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontweight='bold')
    
    # Top right: Average performance
    ax2 = fig.add_subplot(gs[0, 1])
    avg_perf = rankings_df.set_index('model')['avg_gold_performance'].sort_values(ascending=True)
    bars2 = ax2.barh(range(len(avg_perf)), avg_perf.values,
                     color=['#FF6347', '#4169E1', '#2E8B57'])
    ax2.set_yticks(range(len(avg_perf)))
    ax2.set_yticklabels([name.replace('sbert_', '').replace('_', '-') for name in avg_perf.index])
    ax2.set_title('📈 Average Performance', fontweight='bold')
    ax2.set_xlabel('Mean Spearman Correlation')
    
    for bar, value in zip(bars2, avg_perf.values):
        ax2.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontweight='bold')
    
    # Bottom: Recommendation summary
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create recommendation boxes
    recommendations = [
        ("🥇 PRIMARY CHOICE", "all-mpnet-base-v2", "0.763 LaBSE correlation", 
         "Best overall accuracy\nIdeal for critical SEO analysis", '#FFD700'),
        ("🥈 SPEED OPTIMIZED", "all-MiniLM-L6-v2", "0.732 LaBSE correlation", 
         "5x faster inference\nBest for high-volume processing", '#C0C0C0'),
        ("🥉 SPECIALIZED", "gte-base", "0.688 LaBSE correlation", 
         "Domain-specific tasks\nExperimental applications", '#CD7F32')
    ]
    
    for i, (rank, model, corr, desc, color) in enumerate(recommendations):
        x = i * 0.33 + 0.02
        
        # Create recommendation box
        rect = Rectangle((x, 0.3), 0.28, 0.4, facecolor=color, alpha=0.2, 
                        edgecolor=color, linewidth=2)
        ax3.add_patch(rect)
        
        # Add text
        ax3.text(x + 0.14, 0.65, rank, ha='center', va='center', 
                fontsize=12, fontweight='bold')
        ax3.text(x + 0.14, 0.58, model, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        ax3.text(x + 0.14, 0.52, corr, ha='center', va='center', 
                fontsize=10, style='italic')
        ax3.text(x + 0.14, 0.42, desc, ha='center', va='center', 
                fontsize=9, multialignment='center')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('🏆 RECOMMENDATIONS FOR SEO/AIO APPLICATIONS', 
                 fontsize=16, fontweight='bold', y=0.9)
    
    plt.savefig('recommendation_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_scatter_plot_analysis(results_df):
    """Create scatter plot showing correlation vs consistency."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate mean and std for each model
    model_stats = results_df.groupby('model')['spearman_vs_gold'].agg(['mean', 'std']).reset_index()
    
    # Create scatter plot
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, (_, row) in enumerate(model_stats.iterrows()):
        ax.scatter(row['mean'], row['std'], s=200, c=colors[i], alpha=0.7, 
                  edgecolors='black', linewidth=2)
        ax.annotate(row['model'].replace('sbert_', '').replace('_', '-'), 
                   (row['mean'], row['std']), xytext=(10, 10), 
                   textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Average Performance (Mean Spearman Correlation)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Variability (Standard Deviation)', fontsize=12, fontweight='bold')
    ax.set_title('📊 Model Performance vs Consistency\n(Lower right = Better: High performance, Low variability)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add quadrant labels
    ax.axhline(y=model_stats['std'].median(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=model_stats['mean'].median(), color='gray', linestyle='--', alpha=0.5)
    
    ax.text(0.95, 0.95, 'High Perf\nHigh Variability', transform=ax.transAxes, 
            ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
    ax.text(0.95, 0.05, 'High Perf\nLow Variability\n(IDEAL)', transform=ax.transAxes, 
            ha='right', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    ax.text(0.05, 0.95, 'Low Perf\nHigh Variability', transform=ax.transAxes, 
            ha='left', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.5))
    ax.text(0.05, 0.05, 'Low Perf\nLow Variability', transform=ax.transAxes, 
            ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_consistency_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all visualizations."""
    print("🎨 Generating comprehensive visualizations...")
    
    # Load data
    results_df, global_corr_df, rankings_df, relative_perf_df = load_data()
    
    # Generate all charts
    print("📊 1. Creating LaBSE correlation ranking chart...")
    create_labse_correlation_chart(global_corr_df)
    
    print("📊 2. Creating performance heatmap...")
    create_performance_heatmap(results_df)
    
    print("📊 3. Creating correlation matrix...")
    create_correlation_matrix(global_corr_df)
    
    print("📊 4. Creating dataset type comparison...")
    create_dataset_type_comparison(results_df)
    
    print("📊 5. Creating recommendation dashboard...")
    create_recommendation_dashboard(rankings_df, global_corr_df)
    
    print("📊 6. Creating scatter plot analysis...")
    create_scatter_plot_analysis(results_df)
    
    print("\n✅ All visualizations generated successfully!")
    print("\n📁 Generated files:")
    print("   - labse_correlation_ranking.png")
    print("   - performance_heatmap.png") 
    print("   - correlation_matrix.png")
    print("   - dataset_type_comparison.png")
    print("   - recommendation_dashboard.png")
    print("   - performance_consistency_scatter.png")
    
    print("\n🎯 These charts are ready for your research paper!")

if __name__ == "__main__":
    main()

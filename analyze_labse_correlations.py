import pandas as pd
import numpy as np

def analyze_labse_correlations():
    """Analyze all models' correlations with LaBSE specifically."""
    
    # Load the results
    results_df = pd.read_csv("spearman_vs_gold_summary.csv")
    global_corr_df = pd.read_csv("global_model_correlations.csv", index_col=0)
    
    print("=" * 60)
    print("🎯 FOCUSED LaBSE PROXY ANALYSIS")
    print("=" * 60)
    
    # Global correlations with LaBSE
    print("\n📊 GLOBAL CORRELATIONS WITH LaBSE (3,500 text pairs)")
    print("-" * 50)
    
    labse_correlations = global_corr_df['labse'].drop('labse').sort_values(ascending=False)
    
    for i, (model, correlation) in enumerate(labse_correlations.items(), 1):
        stars = "⭐" * min(5, int(correlation * 5))
        print(f"{i}. {model:15s}: {correlation:.3f} {stars}")
    
    # Performance by dataset type
    print("\n📈 PERFORMANCE BY DATASET TYPE")
    print("-" * 50)
    
    # Sentence tasks
    sentence_datasets = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    sentence_results = results_df[results_df['dataset'].isin(sentence_datasets)]
    
    print("\n🔤 SENTENCE-LEVEL TASKS:")
    sentence_avg = sentence_results.groupby('model')['spearman_vs_gold'].mean().sort_values(ascending=False)
    
    for model, avg_score in sentence_avg.items():
        if model != 'labse':
            labse_score = sentence_avg['labse']
            relative_perf = (avg_score / labse_score) * 100 if labse_score != 0 else 0
            print(f"  {model:15s}: {avg_score:.3f} ({relative_perf:+.1f}% vs LaBSE)")
    
    # Word tasks
    word_results = results_df[results_df['dataset'] == 'SimLex-999']
    print("\n📝 WORD-LEVEL TASKS (SimLex-999):")
    
    for _, row in word_results.iterrows():
        if row['model'] != 'labse':
            labse_score = word_results[word_results['model'] == 'labse']['spearman_vs_gold'].iloc[0]
            relative_perf = (row['spearman_vs_gold'] / labse_score) * 100
            print(f"  {row['model']:15s}: {row['spearman_vs_gold']:.3f} ({relative_perf:.1f}% of LaBSE)")
    
    # Recommendations
    print("\n🏆 RECOMMENDATIONS")
    print("-" * 50)
    
    best_proxy = labse_correlations.index[0]
    second_best = labse_correlations.index[1]
    
    print(f"🥇 PRIMARY CHOICE: {best_proxy}")
    print(f"   Global correlation: {labse_correlations[best_proxy]:.3f}")
    print(f"   Use case: Highest accuracy LaBSE replacement")
    
    print(f"\n🥈 SPEED-OPTIMIZED: {second_best}")
    print(f"   Global correlation: {labse_correlations[second_best]:.3f}")
    print(f"   Use case: High-volume processing with minimal accuracy loss")
    
    # Create summary table
    summary_data = []
    for model in labse_correlations.index:
        model_results = results_df[results_df['model'] == model]
        avg_performance = model_results['spearman_vs_gold'].mean()
        std_performance = model_results['spearman_vs_gold'].std()
        
        summary_data.append({
            'model': model,
            'labse_correlation': labse_correlations[model],
            'avg_gold_performance': avg_performance,
            'performance_std': std_performance,
            'rank': list(labse_correlations.index).index(model) + 1
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("labse_proxy_rankings.csv", index=False)
    
    print(f"\n💾 Detailed rankings saved to: labse_proxy_rankings.csv")
    
    # Dataset-specific analysis
    print(f"\n📋 DATASET-SPECIFIC ANALYSIS")
    print("-" * 50)
    
    pivot_table = results_df.pivot(index='dataset', columns='model', values='spearman_vs_gold')
    
    # Calculate relative performance vs LaBSE for each dataset
    relative_performance = pivot_table.copy()
    for dataset in pivot_table.index:
        labse_score = pivot_table.loc[dataset, 'labse']
        for model in pivot_table.columns:
            if model != 'labse' and not pd.isna(labse_score) and labse_score != 0:
                relative_performance.loc[dataset, model] = (pivot_table.loc[dataset, model] / labse_score) * 100
            elif model == 'labse':
                relative_performance.loc[dataset, model] = 100.0
    
    print("\nRelative Performance vs LaBSE (%):")
    print(relative_performance.round(1))
    
    relative_performance.to_csv("relative_performance_vs_labse.csv")
    print(f"\n💾 Relative performance saved to: relative_performance_vs_labse.csv")

if __name__ == "__main__":
    analyze_labse_correlations()

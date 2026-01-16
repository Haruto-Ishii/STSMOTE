import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# モデルリスト
MODELS = ["CatBoost", "XGBoost", "LightGBM", "TabM"]

# 基本指標と向きの定義
BASE_METRICS = {
    "f1_minority": True,
}

# 自動展開
METRIC_DIRECTIONS = {}
for model in MODELS:
    for metric, direction in BASE_METRICS.items():
        METRIC_DIRECTIONS[f"{model}_{metric}"] = direction

def _perform_pairwise_wilcoxon_raw(metric_name, pivot_df, proposed_method, threshold=0.05):

    if proposed_method not in pivot_df.columns:
        return None

    results = []
    
    # 比較対象の手法リスト
    targets = [c for c in pivot_df.columns if c != proposed_method]
    direction = METRIC_DIRECTIONS[metric_name]
    
    for method in targets:
        try:
            # 提案手法と対象手法のデータペア
            x = pivot_df[proposed_method]
            y = pivot_df[method]
            
            diff = x - y
            n_win = (diff > 0).sum() if direction else (diff < 0).sum()
            n_lose = (diff < 0).sum() if direction else (diff > 0).sum()
            
            if np.allclose(x, y):
                p_val = 1.0
                stat = 0.0
            else:
                stat, p_val = wilcoxon(x, y)
            
            results.append({
                "Metric": metric_name,
                "Proposed": proposed_method,
                "Opponent": method,
                "p_value": p_val,
                "Significant": p_val < threshold,
                "Statistic": stat,
                "n_win": n_win,
                "n_lose": n_lose
            })
        except Exception as e:
            results.append({
                "Metric": metric_name,
                "Proposed": proposed_method,
                "Opponent": method,
                "p_value": 1.0,
                "Significant": False,
                "Statistic": np.nan,
                "n_win": 0,
                "n_lose": 0,
                "error": str(e)
            })
            
    return results
        
def _plot_normalized_boxplots(df, metric, output_dir, methods_order, group_cols):
    try:
        def z_score(x):
            if x.std() == 0: return x - x.mean()
            return (x - x.mean()) / x.std()

        df_norm = df.copy()
        norm_col = f"norm_{metric}"
        df_norm[norm_col] = df_norm.groupby(group_cols)[metric].transform(z_score)
        
        median_order = df_norm.groupby("method")[norm_col].median().sort_values(ascending=False).index
        
        plt.figure(figsize=(12, 6))
        
        sns.boxplot(data=df_norm, x="method", y=f"norm_{metric}", showmeans=True, 
                    order=median_order, palette="Set3")
        
        plt.title(f"Normalized Distribution of {metric} (Z-score by Seed)")
        plt.ylabel("Z-score (Relative Performance)")
        plt.xticks(rotation=45)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"boxplot_norm_{metric}.png"))
        plt.close()
        
    except Exception as e:
        print(f"Skipping normalized boxplot for {metric}: {e}")
        plt.close()

def _plot_tradeoff_scatter(df, output_dir, model_name):
    required = [f"{model_name}_min_as_maj", f"{model_name}_maj_as_min"]
    if not all(col in df.columns for col in required): return
    agg_df = df.groupby("method")[required].mean().reset_index()
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=agg_df, x=f"{model_name}_maj_as_min", y=f"{model_name}_min_as_maj", 
                    hue="method", style="method", s=200, palette="deep")
    for i, row in agg_df.iterrows():
        plt.text(row[f"{model_name}_maj_as_min"], row[f"{model_name}_min_as_maj"], f"  {row['method']}", fontsize=9, va='center')
    plt.xlabel("False Positive")
    plt.ylabel("False Negative")
    plt.title(f"False Positive vs False Negative {model_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_tradeoff_{model_name}_FP_FN.png"))
    plt.close()

def run_comparative_analysis(
    cache_manager, 
    methods_to_compare, 
    seed_sizes, 
    seed_undersample_states, 
    minority_class_name,
    output_dir="analysis_report",
    proposed_method="STSMOTE",
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*30}\nStarting Comparative Analysis (Raw p-values)\n{'='*30}")

    df_all = cache_manager.load_results_for_comparison(methods_to_compare, seed_sizes, seed_undersample_states, minority_class_name)
    
    if df_all.empty:
        print("Warning: No result data found.")
        return
    
    available_metrics = [m for m in METRIC_DIRECTIONS.keys() if m in df_all.columns]
    
    if not available_metrics:
        print("Error: No matching metrics found in the loaded data.")
        print("Available columns:", df_all.columns.tolist())
        return

    for col in available_metrics:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    group_cols = ["seed_size", "seed_undersample_state", "random_state", "minority_class_name"]
    rank_df = df_all.copy()
    
    all_stats_results = []
    summary_dfs = []

    print(f"Processing {len(available_metrics)} metrics...")

    for metric in available_metrics:
        higher_is_better = METRIC_DIRECTIONS[metric]
        
        rank_col = f"rank_{metric}"
        rank_df[rank_col] = rank_df.groupby(group_cols)[metric].rank(
            ascending=not higher_is_better, method="average"
        )
        
        agg = rank_df.groupby("method").agg({
            rank_col: ["mean", "std"],
            metric: ["mean", "std"]
        })
        agg.columns = [f"{metric}_{c[1]}" if "rank" not in c[0] else f"Rank_{metric}_{c[1]}" for c in agg.columns]
        summary_dfs.append(agg)

        rank_df["exp_id"] = rank_df[group_cols].astype(str).agg('_'.join, axis=1)
        pivot_score = rank_df.pivot(index="exp_id", columns="method", values=metric)
        
        stats_res = _perform_pairwise_wilcoxon_raw(metric, pivot_score, proposed_method)
        all_stats_results.extend(stats_res)

        _plot_normalized_boxplots(df_all, metric, output_dir, methods_to_compare, group_cols)
        
    def _shorten_name(name):
        return (name.replace("CatBoost", "Cat")
                    .replace("XGBoost", "XGB")
                    .replace("LightGBM", "LGB")
                    .replace("minority", "min")
                    .replace("min_as_maj", "FN")
                    .replace("maj_as_min", "FP"))
        
    if summary_dfs:
        final_summary = pd.concat(summary_dfs, axis=1)
        final_summary.columns = [_shorten_name(c) for c in final_summary.columns]
        summary_path = os.path.join(output_dir, "summary_statistics_all.csv")
        final_summary.to_csv(summary_path)
        print(f"Saved summary statistics to {summary_path}")

    if all_stats_results:
        stats_df = pd.DataFrame(all_stats_results)
        cols = ["Metric", "Proposed", "Opponent", "p_value", "Significant", "n_win", "n_lose", "Statistic"]
        stats_df = stats_df[cols].sort_values(["Metric", "p_value"])
        
        stats_path = os.path.join(output_dir, "wilcoxon_tests_all.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved statistical tests to {stats_path}")
            
    for model_name in MODELS:
        _plot_tradeoff_scatter(df_all, output_dir, model_name)
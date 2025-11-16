import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ==========================
# KONFIGURATION - MODERN DATASETS ONLY
# ==========================

# Max antal par per dataset (för tid och kostnad)
MAX_SAMPLES = {
    "STS-B": 1000,  # Increased since we're using fewer datasets
    "SimLex-999": 999,  # Use full dataset
}

# Sentence-Transformer-modeller (endast engelska här)
SENTENCE_MODELS = {
    "sbert_mpnet": "sentence-transformers/all-mpnet-base-v2",
    "sbert_minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "labse": "sentence-transformers/LaBSE",
    "gte_base": "thenlper/gte-base",
}

BATCH_SIZE = 64


# ==========================
# HJÄLPMETODER
# ==========================

def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Beräknar cosine similarity radvis mellan två matriser a och b
    med shape (n, d).
    """
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)


# ==========================
# LADDA DATASETS - MODERN ONLY
# ==========================

def _normalize_and_sample(df: pd.DataFrame,
                          dataset_name: str,
                          pair_type: str,
                          max_samples: Optional[int]) -> pd.DataFrame:
    """
    Hjälpfunktion: sätter dataset/type, samplar max_samples.
    """
    df["dataset"] = dataset_name
    df["type"] = pair_type
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return df


def load_stsb(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    STS-Benchmark (engelska meningspar, gold-score 0–1).
    Källa: sentence-transformers/stsb.
    """
    print("  Loading STS-Benchmark dataset...")
    ds = load_dataset("sentence-transformers/stsb")
    frames = []
    for split in ds.keys():  # train / validation / test
        df_split = ds[split].to_pandas()
        frames.append(
            df_split[["sentence1", "sentence2", "score"]].rename(
                columns={
                    "sentence1": "text1",
                    "sentence2": "text2",
                    "score": "gold_score",
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return _normalize_and_sample(df, "STS-B", "sentence", max_samples)


def load_simlex(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    SimLex-999 (engelska ordpar, gold-score ~0–10).
    Källa: tasksource/simlex.

    Returnerar text1/text2 och gold_score normaliserad till 0–1.
    """
    print("  Loading SimLex-999 dataset...")
    ds = load_dataset("tasksource/simlex")
    df = ds["train"].to_pandas()

    # Försök hitta ordkolumner
    col_map = {c.lower(): c for c in df.columns}
    word1_col = col_map.get("word1")
    word2_col = col_map.get("word2")
    if word1_col is None or word2_col is None:
        str_cols = [c for c in df.columns if df[c].dtype == "object"]
        if len(str_cols) < 2:
            raise ValueError("Hittar inte word1/word2 i SimLex-999.")
        word1_col, word2_col = str_cols[:2]

    # Försök hitta scorekolumn
    score_col = None
    for cand in ["simlex999", "similarity", "score"]:
        if cand in col_map:
            score_col = col_map[cand]
            break
    
    # Check for exact column name "SimLex999"
    if score_col is None and "SimLex999" in df.columns:
        score_col = "SimLex999"

    out = pd.DataFrame()
    out["text1"] = df[word1_col].astype(str)
    out["text2"] = df[word2_col].astype(str)

    if score_col is not None:
        scores = df[score_col].astype(float)
        max_val = scores.max()
        out["gold_score"] = scores / max_val if max_val > 0 else scores
    else:
        out["gold_score"] = np.nan

    return _normalize_and_sample(out, "SimLex-999", "word", max_samples)


def load_all_datasets() -> pd.DataFrame:
    """
    Laddar endast moderna datasets (STS-B och SimLex-999).
    """
    print("🔄 Loading MODERN datasets only (excluding STS 2012-2016)...")
    print("=" * 60)
    
    print("📝 Loading STS-B (sentence similarity)...")
    stsb_df = load_stsb(MAX_SAMPLES["STS-B"])
    print(f"  ✅ {len(stsb_df)} sentence pairs loaded")

    print("\n📝 Loading SimLex-999 (word similarity)...")
    simlex_df = load_simlex(MAX_SAMPLES["SimLex-999"])
    print(f"  ✅ {len(simlex_df)} word pairs loaded")

    all_df = pd.concat([stsb_df, simlex_df], ignore_index=True)
    
    print(f"\n📊 TOTAL: {len(all_df)} text pairs from {len(all_df['dataset'].unique())} modern datasets")
    print("=" * 60)
    return all_df


# ==========================
# EMBEDDING-MODELLER
# ==========================

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        return np.array(
            self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        )


# ==========================
# KÖRNING / JÄMFÖRELSE
# ==========================

def compute_model_similarities_for_df(
    df: pd.DataFrame,
    embedder,
    model_name: str,
) -> np.ndarray:
    """
    Beräknar embeddings och cosine similarity för text1/text2 i df.
    Returnerar en vektor med likheter (same order som df).
    """
    texts1 = df["text1"].tolist()
    texts2 = df["text2"].tolist()

    print(f"    Computing embeddings for {len(texts1)} text pairs...")
    emb1 = embedder.encode(texts1)
    emb2 = embedder.encode(texts2)

    sims = cosine_similarity_batch(emb1, emb2)
    df[f"sim_{model_name}"] = sims
    return sims


def evaluate_against_gold(
    gold: np.ndarray,
    sims: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Spearman-korrelation mellan gold-score och model-similarity.
    """
    results = {}
    mask = ~np.isnan(gold)
    if mask.sum() == 0:
        return results

    gold_valid = gold[mask]
    for model_name, sim_vec in sims.items():
        sim_valid = sim_vec[mask]
        rho, _ = spearmanr(gold_valid, sim_valid)
        results[model_name] = rho
    return results


def evaluate_model_correlations(
    sims: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Spearman-korrelation mellan modellerna (sim-vektorerna).
    Returnerar en kvadratisk DataFrame.
    """
    model_names = list(sims.keys())
    data = np.zeros((len(model_names), len(model_names)))
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                data[i, j] = 1.0
            else:
                rho, _ = spearmanr(sims[m1], sims[m2])
                data[i, j] = rho
    return pd.DataFrame(data, index=model_names, columns=model_names)


def main():
    print("🎯 MODERN DATASETS ONLY - LaBSE PROXY ANALYSIS")
    print("🚫 Excluding: STS 2012, 2013, 2014, 2015, 2016")
    print("✅ Including: STS-B, SimLex-999")
    print()
    
    # ---- Ladda moderna datasets ----
    all_df = load_all_datasets()

    # ---- Initiera modeller ----
    print("\n🤖 Initializing embedding models...")
    st_embedders = {
        name: SentenceTransformerEmbedder(model_name)
        for name, model_name in SENTENCE_MODELS.items()
    }

    # För global analys av modellkorrelationer
    global_sims: Dict[str, List[np.ndarray]] = {
        name: [] for name in SENTENCE_MODELS.keys()
    }

    summary_rows = []

    # ---- Kör per dataset ----
    print("\n" + "="*60)
    print("🔄 RUNNING ANALYSIS PER DATASET")
    print("="*60)
    
    for dataset_name in sorted(all_df["dataset"].unique()):
        subset = all_df[all_df["dataset"] == dataset_name].reset_index(drop=True)
        pair_type = subset["type"].iloc[0]
        print(f"\n📊 Dataset: {dataset_name} ({pair_type})")
        print(f"   Pairs: {len(subset)}")

        sims: Dict[str, np.ndarray] = {}

        # Sentence-transformer-modeller
        for short_name, embedder in st_embedders.items():
            print(f"  🔄 Processing {short_name}...")
            sims[short_name] = compute_model_similarities_for_df(
                subset.copy(), embedder, short_name
            )
            global_sims[short_name].append(sims[short_name])

        gold = subset["gold_score"].to_numpy(dtype=float)

        # 1) Korrelation mot gold (om finns)
        gold_corr = evaluate_against_gold(gold, sims)
        if gold_corr:
            print(f"\n  📈 Spearman correlation vs gold standard:")
            for model_name, rho in gold_corr.items():
                emoji = "🥇" if model_name == max(gold_corr, key=gold_corr.get) else "📊"
                print(f"    {emoji} {model_name:12s}: {rho: .4f}")
        else:
            print(f"\n  ⚠️  No gold scores available for {dataset_name}")

        # 2) Modeller mot varandra (inom dataset)
        corr_matrix = evaluate_model_correlations(sims)
        print(f"\n  🔗 Inter-model correlations:")
        print(corr_matrix.round(4))

        # Lägg till i summary (per dataset)
        for model_name, rho in gold_corr.items():
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "type": pair_type,
                    "model": model_name,
                    "spearman_vs_gold": rho,
                }
            )

        # Spara korrelationsmatrisen till CSV för detta dataset
        corr_matrix.to_csv(f"modern_model_correlations_{dataset_name}.csv")

    # ---- Global analys: vilken modell liknar LaBSE mest? ----
    print("\n" + "="*60)
    print("🎯 GLOBAL LaBSE PROXY ANALYSIS (MODERN DATASETS)")
    print("="*60)

    # Konkatenara sims över datasets
    concat_sims = {
        name: np.concatenate(vs) for name, vs in global_sims.items()
    }

    anchor = "labse"
    print(f"\n🎯 Correlation with LaBSE (across {len(concat_sims[anchor])} pairs):")
    anchor_vec = concat_sims[anchor]
    rows = []
    for model_name, vec in concat_sims.items():
        if model_name == anchor:
            continue
        rho, _ = spearmanr(anchor_vec, vec)
        rows.append((model_name, rho))
    
    # sortera på rho
    rows.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 RANKING:")
    for i, (model_name, rho) in enumerate(rows, 1):
        if i == 1:
            emoji = "🥇"
            badge = "PRIMARY CHOICE"
        elif i == 2:
            emoji = "🥈"
            badge = "SPEED OPTIMIZED"
        else:
            emoji = "🥉"
            badge = "SPECIALIZED"
        
        print(f"  {emoji} {i}. {model_name:12s}: {rho: .4f} ({badge})")

    # Global korrelationsmatris
    global_corr_matrix = evaluate_model_correlations(concat_sims)
    print(f"\n🔗 Global correlation matrix ({len(concat_sims[anchor])} pairs):")
    print(global_corr_matrix.round(4))
    global_corr_matrix.to_csv("modern_global_model_correlations.csv")

    # ---- Summering mot gold-scores ----
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(f"\n📊 DETAILED RESULTS:")
        print(summary_df.round(4))
        summary_df.to_csv("modern_spearman_vs_gold_summary.csv", index=False)
        
        # Genomsnittlig prestanda per modell
        avg_performance = summary_df.groupby('model')['spearman_vs_gold'].agg(['mean', 'std']).round(4)
        print(f"\n📈 AVERAGE PERFORMANCE PER MODEL:")
        print(avg_performance)
        avg_performance.to_csv("modern_average_model_performance.csv")
        
        # LaBSE proxy rankings
        labse_correlations = global_corr_matrix['labse'].drop('labse').sort_values(ascending=False)
        proxy_rankings = []
        for i, (model, correlation) in enumerate(labse_correlations.items(), 1):
            model_results = summary_df[summary_df['model'] == model]
            avg_perf = model_results['spearman_vs_gold'].mean()
            std_perf = model_results['spearman_vs_gold'].std()
            
            proxy_rankings.append({
                'rank': i,
                'model': model,
                'labse_correlation': correlation,
                'avg_gold_performance': avg_perf,
                'performance_std': std_perf
            })
        
        proxy_df = pd.DataFrame(proxy_rankings)
        proxy_df.to_csv("modern_labse_proxy_rankings.csv", index=False)
        
    else:
        print("\n⚠️  No datasets had gold scores – no gold correlations calculated.")

    print(f"\n" + "="*60)
    print("✅ MODERN DATASETS ANALYSIS COMPLETE!")
    print("="*60)
    print("📁 Files created:")
    print("   - modern_spearman_vs_gold_summary.csv")
    print("   - modern_average_model_performance.csv") 
    print("   - modern_global_model_correlations.csv")
    print("   - modern_labse_proxy_rankings.csv")
    print("   - modern_model_correlations_[dataset].csv")
    print(f"\n🎯 Focus: {len(concat_sims[anchor])} pairs from modern datasets only")


if __name__ == "__main__":
    main()

import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==========================
# KONFIGURATION - REDUCED FOR TESTING
# ==========================

# Max antal par per dataset (för tid och kostnad)
MAX_SAMPLES = {
    "STS-B": 100,  # Reduced for testing
}

# Only test with one smaller model first
SENTENCE_MODELS = {
    "sbert_minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

# OpenAI-embedding-modell
OPENAI_MODEL_NAME = "text-embedding-3-large"

BATCH_SIZE = 32  # Smaller batch size


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
# LADDA DATASETS
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


def load_all_datasets() -> pd.DataFrame:
    """
    Laddar alla valda datasets och returnerar en gemensam DataFrame.
    """
    print("Laddar STS-B (meningspar)...")
    stsb_df = load_stsb(MAX_SAMPLES["STS-B"])
    print(f"  -> {len(stsb_df)} par")

    print(f"\nTotalt antal par (alla dataset): {len(stsb_df)}")
    return stsb_df


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


class OpenAIEmbedder:
    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name

    def encode(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI batches"):
            batch = texts[i: i + batch_size]
            try:
                resp = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                batch_emb = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
                all_embeddings.extend(batch_emb)
            except Exception as e:
                print(f"OpenAI API Error: {e}")
                print("Please ensure you have set your OPENAI_API_KEY environment variable")
                raise
        return np.vstack(all_embeddings)


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
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("Continuing with only Sentence Transformer models...")
        use_openai = False
    else:
        print("OpenAI API key found!")
        use_openai = True

    # ---- Ladda alla datasets ----
    all_df = load_all_datasets()

    # ---- Initiera modeller ----
    print("\nInitierar Sentence-Transformer-modeller...")
    st_embedders = {
        name: SentenceTransformerEmbedder(model_name)
        for name, model_name in SENTENCE_MODELS.items()
    }

    if use_openai:
        print("Initierar OpenAI-klient...")
        openai_embedder = OpenAIEmbedder(OPENAI_MODEL_NAME)

    summary_rows = []

    # ---- Kör per dataset ----
    for dataset_name in sorted(all_df["dataset"].unique()):
        subset = all_df[all_df["dataset"] == dataset_name].reset_index(drop=True)
        pair_type = subset["type"].iloc[0]
        print(f"\n=== Dataset: {dataset_name} ({pair_type}) ===")
        print(f"Antal par: {len(subset)}")

        sims: Dict[str, np.ndarray] = {}

        # Sentence-transformer-modeller
        for short_name, embedder in st_embedders.items():
            print(f"  -> {short_name} ...")
            sims[short_name] = compute_model_similarities_for_df(
                subset.copy(), embedder, short_name
            )

        # OpenAI
        if use_openai:
            print("  -> openai (text-embedding-3-large)...")
            sims["openai"] = compute_model_similarities_for_df(
                subset.copy(), openai_embedder, "openai"
            )

        gold = subset["gold_score"].to_numpy(dtype=float)

        # 1) Korrelation mot gold (om finns)
        gold_corr = evaluate_against_gold(gold, sims)
        if gold_corr:
            print("\n  Spearman mot gold_score:")
            for model_name, rho in gold_corr.items():
                print(f"    {model_name:12s}: {rho: .4f}")
        else:
            print("\n  Inga gold-scores för detta dataset (hoppar över gold-jämförelse).")

        # 2) Modeller mot varandra (inom dataset)
        if len(sims) > 1:
            corr_matrix = evaluate_model_correlations(sims)
            print("\n  Spearman mellan modellerna (inom dataset):")
            print(corr_matrix)

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

    # ---- Summering mot gold-scores ----
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n\n=== Översikt: Spearman mot gold per dataset/modell ===")
        print(summary_df)
        summary_df.to_csv("test_spearman_vs_gold_summary.csv", index=False)
        print("Results saved to test_spearman_vs_gold_summary.csv")
    else:
        print("\nIngen dataset hade gold-scores – inga gold-korrelationer beräknade.")


if __name__ == "__main__":
    main()

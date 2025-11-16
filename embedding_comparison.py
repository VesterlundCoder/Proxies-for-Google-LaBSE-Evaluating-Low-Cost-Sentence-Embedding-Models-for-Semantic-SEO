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
# KONFIGURATION
# ==========================

# Max antal par per dataset (för tid och kostnad)
MAX_SAMPLES = {
    "STS-B": 500,
    "SimLex-999": 500,
    "STS12": 500,
    "STS13": 500,
    "STS14": 500,
    "STS15": 500,
    "STS16": 500,
}

# Sentence-Transformer-modeller (endast engelska här)
SENTENCE_MODELS = {
    "sbert_mpnet": "sentence-transformers/all-mpnet-base-v2",
    "sbert_minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "labse": "sentence-transformers/LaBSE",
    "gte_base": "thenlper/gte-base",
}

# OpenAI-embedding-modell
OPENAI_MODEL_NAME = "text-embedding-3-large"

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


def load_simlex(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    SimLex-999 (engelska ordpar, gold-score ~0–10).
    Källa: tasksource/simlex.

    Returnerar text1/text2 och gold_score normaliserad till 0–1.
    """
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


def load_mteb_sts(dataset_id: str,
                  pretty_name: str,
                  max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Generisk loader för MTEB STS-dataset (STS12–STS16).
    Antagande: datasetet har minst två textkolumner + minst en numerisk scorekolumn
    (0–5 eller liknande) som normaliseras till 0–1.
    """
    ds = load_dataset(dataset_id)
    frames = []

    for split in ds.keys():  # t.ex. train/test
        df_split = ds[split].to_pandas()

        # Hitta två textkolumner
        str_cols = [c for c in df_split.columns if df_split[c].dtype == "object"]
        if len(str_cols) < 2:
            raise ValueError(f"Hittar inte två textkolumner i {dataset_id} ({split}).")

        text1_col, text2_col = str_cols[:2]

        # Hitta någon numerisk scorekolumn
        num_cols = [c for c in df_split.columns
                    if np.issubdtype(df_split[c].dtype, np.number)]
        if num_cols:
            score_col = num_cols[0]
            scores = df_split[score_col].astype(float)
            max_val = scores.max()
            gold = scores / max_val if max_val > 0 else scores
        else:
            gold = np.full(len(df_split), np.nan, dtype=float)

        frames.append(
            pd.DataFrame(
                {
                    "text1": df_split[text1_col].astype(str),
                    "text2": df_split[text2_col].astype(str),
                    "gold_score": gold,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    return _normalize_and_sample(df, pretty_name, "sentence", max_samples)


def load_all_datasets() -> pd.DataFrame:
    """
    Laddar alla valda datasets och returnerar en gemensam DataFrame.
    """
    print("Laddar STS-B (meningspar)...")
    stsb_df = load_stsb(MAX_SAMPLES["STS-B"])
    print(f"  -> {len(stsb_df)} par")

    print("Laddar SimLex-999 (ordpar)...")
    simlex_df = load_simlex(MAX_SAMPLES["SimLex-999"])
    print(f"  -> {len(simlex_df)} par")

    print("Laddar MTEB STS12–STS16...")
    sts12_df = load_mteb_sts("mteb/sts12-sts", "STS12", MAX_SAMPLES["STS12"])
    sts13_df = load_mteb_sts("mteb/sts13-sts", "STS13", MAX_SAMPLES["STS13"])
    sts14_df = load_mteb_sts("mteb/sts14-sts", "STS14", MAX_SAMPLES["STS14"])
    sts15_df = load_mteb_sts("mteb/sts15-sts", "STS15", MAX_SAMPLES["STS15"])
    sts16_df = load_mteb_sts("mteb/sts16-sts", "STS16", MAX_SAMPLES["STS16"])

    for df, name in [
        (sts12_df, "STS12"),
        (sts13_df, "STS13"),
        (sts14_df, "STS14"),
        (sts15_df, "STS15"),
        (sts16_df, "STS16"),
    ]:
        print(f"  -> {name}: {len(df)} par")

    all_df = pd.concat(
        [stsb_df, simlex_df, sts12_df, sts13_df, sts14_df, sts15_df, sts16_df],
        ignore_index=True,
    )
    print(f"\nTotalt antal par (alla dataset): {len(all_df)}")
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
                show_progress_bar=False,
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
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_emb = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
            all_embeddings.extend(batch_emb)
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
    # ---- Ladda alla datasets ----
    all_df = load_all_datasets()

    # ---- Initiera modeller ----
    print("\nInitierar Sentence-Transformer-modeller...")
    st_embedders = {
        name: SentenceTransformerEmbedder(model_name)
        for name, model_name in SENTENCE_MODELS.items()
    }

    print("Initierar OpenAI-klient...")
    openai_embedder = OpenAIEmbedder(OPENAI_MODEL_NAME)

    # För global analys av modellkorrelationer
    global_sims: Dict[str, List[np.ndarray]] = {
        **{name: [] for name in SENTENCE_MODELS.keys()},
        "openai": [],
    }

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
            global_sims[short_name].append(sims[short_name])

        # OpenAI
        print("  -> openai (text-embedding-3-large)...")
        sims["openai"] = compute_model_similarities_for_df(
            subset.copy(), openai_embedder, "openai"
        )
        global_sims["openai"].append(sims["openai"])

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

        # Spara korrelationsmatrisen till CSV för detta dataset
        corr_matrix.to_csv(f"model_correlations_{dataset_name}.csv")

    # ---- Global analys: vilken modell liknar LaBSE / OpenAI mest? ----
    print("\n\n=== Global modellkorrelation (alla dataset ihop) ===")

    # Konkatenara sims över datasets
    concat_sims = {
        name: np.concatenate(vs) for name, vs in global_sims.items()
    }

    anchors = ["labse", "openai"]
    for anchor in anchors:
        print(f"\n>> Korrelation mot ankarmodell: {anchor}")
        anchor_vec = concat_sims[anchor]
        rows = []
        for model_name, vec in concat_sims.items():
            if model_name == anchor:
                continue
            rho, _ = spearmanr(anchor_vec, vec)
            rows.append((model_name, rho))
        # sortera på rho
        rows.sort(key=lambda x: x[1], reverse=True)
        for model_name, rho in rows:
            print(f"  {model_name:12s}: {rho: .4f}")

    # ---- Summering mot gold-scores ----
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n\n=== Översikt: Spearman mot gold per dataset/modell ===")
        print(summary_df)
        summary_df.to_csv("spearman_vs_gold_summary.csv", index=False)
    else:
        print("\nIngen dataset hade gold-scores – inga gold-korrelationer beräknade.")


if __name__ == "__main__":
    main()
